/**
 * Pure JavaScript LLM (decoder-only Transformer)
 * -------------------------------------------------------------
 * - 8 attention heads, d_model = 512, FFN = 4x (2048)
 * - Word-level tokenizer (simple, whitespace + punctuation split)
 * - Causal self-attention with learned token + position embeddings
 * - Cross-entropy loss
 * - Adam optimizer (no external libs)
 * - Full backprop through attention, MLP, layernorm, embeddings
 * - Save/Load weights and optimizer state to JSON (per checkpoint)
 *
 * NOTE
 * ----
 * This file targets **Node.js** (uses `fs`). No third-party libraries.
 * It is educational and optimized for clarity, not speed. Training at
 * d_model=512 can be slow in pure JS—start with small batches/seq.
 *
 * HOW TO USE
 * ----------
 * 1) Put a small training corpus into `corpusText` below (or load from file).
 * 2) Run: `node llm.js` (rename file if needed).
 * 3) The script will build a word-level vocab, train a few steps,
 *    save weights to `weights.json` and optimizer state to `opt.json`.
 * 4) Use `generate()` at the end to sample next-word predictions.
 *
 * You can tweak HYPS below. Defaults honor your request:
 *   HEADS=8, D_MODEL=512, FFN_HIDDEN=2048
 */

const fs = require('fs');

/********************\
|* Hyperparameters  *|
\********************/
const D_MODEL = 512;              // embedding size
const N_HEADS = 8;                // attention heads
const HEAD_DIM = D_MODEL / N_HEADS; // 64
if (HEAD_DIM !== Math.floor(HEAD_DIM)) throw new Error('D_MODEL must be divisible by N_HEADS');
const FFN_HIDDEN = 4 * D_MODEL;   // 2048
const N_LAYERS = 2;               // you can increase to 4/6/8 later
const MAX_SEQ_LEN = 64;           // keep modest to start
const BATCH_SIZE = 2;             // pure JS -> keep tiny
const LR = 2e-4;                  // Adam learning rate
const BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
const WEIGHT_INIT_SCALE = 0.02;   // small init like GPT
const DROPOUT_P = 0.0;            // set >0.0 if you want regularization (train only)
const CKPT_WEIGHTS = 'weights.json';
const CKPT_OPT = 'opt.json';

/********************\
|* Utility helpers  *|
\********************/
function randn() { // Box-Muller
  let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function zeros(n) { return new Float64Array(n); }
function randnArray(n, scale=1.0) { const a = new Float64Array(n); for (let i=0;i<n;i++) a[i]=randn()*scale; return a; }
function randUniform(n, scale=1.0) { const a = new Float64Array(n); for (let i=0;i<n;i++) a[i]=(Math.random()*2-1)*scale; return a; }
function clipInPlace(a, bound=5.0){ for(let i=0;i<a.length;i++){ if(a[i]>bound) a[i]=bound; if(a[i]<-bound) a[i]=-bound; } return a; }

function argmax(arr){ let m=-Infinity, idx=-1; for(let i=0;i<arr.length;i++){ if(arr[i]>m){ m=arr[i]; idx=i; } } return idx; }
function softmaxInPlace(x){ let m=-Infinity; for (let i=0;i<x.length;i++) if (x[i]>m) m=x[i]; let s=0; for(let i=0;i<x.length;i++){ x[i]=Math.exp(x[i]-m); s+=x[i]; } for(let i=0;i<x.length;i++) x[i]/=s; return x; }

/*********************\
|* Tokenizer (words) *|
\*********************/
class WordTokenizer {
  constructor() { this.stoi = {}; this.itos = []; this.vocabBuilt=false; }
  static split(text){
    // split on non-letters/digits underscores, keep apostrophes in words
    return text.toLowerCase().split(/[^a-z0-9']+/).filter(w=>w.length>0);
  }
  build(text){
    const words = WordTokenizer.split(text);
    const uniq = new Map();
    for (const w of words) { if (!uniq.has(w)) uniq.set(w, 1); else uniq.set(w, uniq.get(w)+1); }
    const sorted = Array.from(uniq.keys()).sort();
    this.itos = ['<pad>','<bos>','<eos>','<unk>', ...sorted];
    this.stoi = Object.fromEntries(this.itos.map((w,i)=>[w,i]));
    this.vocabBuilt=true;
  }
  encode(text, addBosEos=true){
    if(!this.vocabBuilt) throw new Error('Tokenizer not built');
    const toks = WordTokenizer.split(text).map(w=>this.stoi[w] ?? this.stoi['<unk>']);
    if(addBosEos) return [this.stoi['<bos>'], ...toks, this.stoi['<eos>']];
    return toks;
  }
  decode(ids){ return ids.map(i=>this.itos[i] ?? '<unk>').join(' '); }
  size(){ return this.itos.length; }
}

/*******************\
|* Tensor helpers  *|
\*******************/
// Represent 2D matrices as {rows, cols, data: Float64Array}
function mat(rows, cols, data=null){ return {rows, cols, data: data? data: zeros(rows*cols)}; }
function matCopy(A){ return {rows:A.rows, cols:A.cols, data: new Float64Array(A.data)}; }
function get(A,i,j){ return A.data[i*A.cols+j]; }
function set(A,i,j,v){ A.data[i*A.cols+j]=v; }
function fillRandn(A, scale){ for(let i=0;i<A.data.length;i++) A.data[i]=randn()*scale; return A; }
function fillZero(A){ A.data.fill(0); return A; }
function addInPlace(A,B){ for(let i=0;i<A.data.length;i++) A.data[i]+=B.data[i]; return A; }
function addScalarInPlace(A,s){ for(let i=0;i<A.data.length;i++) A.data[i]+=s; return A; }
function mulScalarInPlace(A,s){ for(let i=0;i<A.data.length;i++) A.data[i]*=s; return A; }
function matmul(A,B){ // (m x n)*(n x p) -> (m x p)
  if (A.cols!==B.rows) throw new Error('matmul shape mismatch');
  const C = mat(A.rows, B.cols);
  const M=A.rows, N=A.cols, P=B.cols;
  const Ad=A.data, Bd=B.data, Cd=C.data;
  for(let i=0;i<M;i++){
    for(let k=0;k<N;k++){
      const a = Ad[i*N+k];
      const bk = k*P;
      for(let j=0;j<P;j++){
        Cd[i*P+j] += a * Bd[bk+j];
      }
    }
  }
  return C;
}
function transpose(A){ const T = mat(A.cols, A.rows);
  for(let i=0;i<A.rows;i++) for(let j=0;j<A.cols;j++) set(T,j,i,get(A,i,j));
  return T; }
function add(A,B){ const C=mat(A.rows,A.cols); for(let i=0;i<A.data.length;i++) C.data[i]=A.data[i]+B.data[i]; return C; }
function hadamard(A,B){ const C=mat(A.rows,A.cols); for(let i=0;i<A.data.length;i++) C.data[i]=A.data[i]*B.data[i]; return C; }
function concatCols(A,B){ if(A.rows!==B.rows) throw new Error('concat rows mismatch');
  const C = mat(A.rows, A.cols+B.cols);
  for(let i=0;i<A.rows;i++){
    for(let j=0;j<A.cols;j++) set(C,i,j,get(A,i,j));
    for(let j=0;j<B.cols;j++) set(C,i,A.cols+j,get(B,i,j));
  }
  return C; }

/*******************\n|* Layers & Grads *|\n*******************/
function kaimingInit(rows, cols, scale=WEIGHT_INIT_SCALE){
  // simple scaled normal
  return fillRandn(mat(rows, cols), scale);
}

class Linear {
  constructor(inF, outF){
    this.W = kaimingInit(inF, outF);
    this.b = mat(1, outF);
    fillZero(this.b);
    // grads
    this.gW = mat(inF, outF);
    this.gb = mat(1, outF);
  }
  forward(X){ // (B*T, inF)
    this.Xcache = X; // save for backward
    const Y = add(matmul(X, this.W), this.bRepeat(X.rows));
    return Y;
  }
  bRepeat(n){ const Bmat = mat(n, this.b.cols); for(let i=0;i<n;i++) for(let j=0;j<this.b.cols;j++) set(Bmat,i,j, this.b.data[j]); return Bmat; }
  backward(dY){ // dL/dX, and accumulate grads on W,b
    // dW = X^T * dY
    const Xt = transpose(this.Xcache);
    const dW = matmul(Xt, dY);
    addInPlace(this.gW, dW);
    // db = row-sum of dY
    for(let j=0;j<this.gb.cols;j++){
      let s=0; for(let i=0;i<dY.rows;i++) s += get(dY,i,j);
      this.gb.data[j] += s;
    }
    // dX = dY * W^T
    const WT = transpose(this.W);
    const dX = matmul(dY, WT);
    return dX;
  }
  step(opt, t){ opt.adamStep(this.W, this.gW, this.b, this.gb, t); this.zeroGrad(); }
  zeroGrad(){ fillZero(this.gW); fillZero(this.gb); }
  params(){ return [{name:'W', M:this.W},{name:'b', M:this.b}]; }
}

class LayerNorm {
  constructor(dim, eps=1e-5){
    this.eps=eps; this.dim=dim;
    this.gamma = mat(1, dim); this.beta = mat(1, dim);
    for(let j=0;j<dim;j++){ this.gamma.data[j]=1.0; this.beta.data[j]=0.0; }
    this.ggamma = mat(1, dim); this.gbeta = mat(1, dim);
  }
  forward(X){ // per row normalization
    const B=X.rows, D=X.cols; this.X=X;
    this.mean = zeros(B); this.varr = zeros(B);
    const out = mat(B,D);
    for(let i=0;i<B;i++){
      let m=0; for(let j=0;j<D;j++) m+=get(X,i,j); m/=D; this.mean[i]=m;
      let v=0; for(let j=0;j<D;j++){ const d=get(X,i,j)-m; v+=d*d; } v/=D; this.varr[i]=v;
      const inv = 1/Math.sqrt(v+this.eps);
      for(let j=0;j<D;j++){
        const xhat = (get(X,i,j)-m)*inv;
        const y = xhat*this.gamma.data[j] + this.beta.data[j];
        set(out,i,j,y);
      }
    }
    return out;
  }
  backward(dY){
    const B=this.X.rows, D=this.X.cols; const dX = mat(B,D);
    // accumulate ggamma & gbeta
    for(let j=0;j<D;j++){
      let gg=0, gb=0; for(let i=0;i<B;i++){
        const inv = 1/Math.sqrt(this.varr[i]+this.eps);
        const xhat = (get(this.X,i,j)-this.mean[i])*inv;
        gg += get(dY,i,j)*xhat; gb += get(dY,i,j);
      }
      this.ggamma.data[j]+=gg; this.gbeta.data[j]+=gb;
    }
    // dX formula for LayerNorm
    for(let i=0;i<B;i++){
      const m=this.mean[i], v=this.varr[i]; const inv=1/Math.sqrt(v+this.eps);
      let sum_dy=0, sum_dy_xhat=0;
      for(let j=0;j<D;j++){
        const xhat=(get(this.X,i,j)-m)*inv;
        const dy = get(dY,i,j)*this.gamma.data[j];
        sum_dy += dy; sum_dy_xhat += dy*xhat;
      }
      for(let j=0;j<D;j++){
        const xhat=(get(this.X,i,j)-m)*inv;
        const dy = get(dY,i,j)*this.gamma.data[j];
        const term1 = (dy*D - sum_dy - xhat*sum_dy_xhat) * inv / D;
        set(dX,i,j, term1);
      }
    }
    return dX;
  }
  step(opt,t){ opt.adamStepSimple(this.gamma, this.ggamma, this.beta, this.gbeta, t); this.zeroGrad(); }
  zeroGrad(){ fillZero(this.ggamma); fillZero(this.gbeta); }
  params(){ return [{name:'gamma', M:this.gamma},{name:'beta', M:this.beta}]; }
}

function geluInPlace(A){ // approximate GELU
  for(let i=0;i<A.data.length;i++){
    const x=A.data[i];
    A.data[i]=0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3))));
  }
  return A;
}
function geluBackward(dY, Xpre){ // derivative wrt pre-activation Xpre
  const B=dY.rows, D=dY.cols; const dX=mat(B,D);
  for(let i=0;i<B;i++) for(let j=0;j<D;j++){
    const x = get(Xpre,i,j);
    const tanhArg = Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x);
    const th = Math.tanh(tanhArg);
    const sech2 = 1 - th*th;
    const geluPrime = 0.5*(1+th) + 0.5*x*sech2*Math.sqrt(2/Math.PI)*(1+3*0.044715*x*x);
    set(dX,i,j, get(dY,i,j)*geluPrime);
  }
  return dX;
}

class FeedForward {
  constructor(dModel, hidden){ this.l1=new Linear(dModel, hidden); this.l2=new Linear(hidden, dModel); }
  forward(X){ this.Xpre = this.l1.forward(X); this.Xact = geluInPlace(matCopy(this.Xpre)); const Y=this.l2.forward(this.Xact); return Y; }
  backward(dY){ const dX2 = this.l2.backward(dY); const dAct = geluBackward(dX2, this.Xpre); const dX1 = this.l1.backward(dAct); return dX1; }
  step(opt,t){ this.l1.step(opt,t); this.l2.step(opt,t); }
  zeroGrad(){ this.l1.zeroGrad(); this.l2.zeroGrad(); }
  params(){ return [{name:'l1W',M:this.l1.W},{name:'l1b',M:this.l1.b},{name:'l2W',M:this.l2.W},{name:'l2b',M:this.l2.b}]; }
}

class MultiHeadAttention {
  constructor(dModel, nHeads){
    this.dModel=dModel; this.nHeads=nHeads; this.headDim = dModel/nHeads;
    this.Wq=new Linear(dModel,dModel); this.Wk=new Linear(dModel,dModel); this.Wv=new Linear(dModel,dModel); this.Wo=new Linear(dModel,dModel);
  }
  splitHeads(X){ // (B*T, D) -> array of heads each (B*T, headDim)
    const heads=[]; const D=X.cols; const H=this.nHeads; const Hd=this.headDim;
    for(let h=0; h<H; h++){
      const start=h*Hd; const Mh=mat(X.rows,Hd);
      for(let i=0;i<X.rows;i++) for(let j=0;j<Hd;j++) set(Mh,i,j,get(X,i,start+j));
      heads.push(Mh);
    }
    return heads;
  }
  concatHeads(heads){ // array -> (B*T, D)
    const B=heads[0].rows, Hd=heads[0].cols, H=heads.length; const M=mat(B,H*Hd);
    for(let h=0;h<H;h++) for(let i=0;i<B;i++) for(let j=0;j<Hd;j++) set(M,i,h*Hd+j, get(heads[h],i,j));
    return M;
  }
  forward(Xseq, T){
    // Xseq is (B*T, D). We'll reshape via indices to perform attention over each sequence separately.
    this.BT = Xseq.rows; this.D = Xseq.cols; this.T=T; // store
    // linear projections
    const Qbig=this.Wq.forward(Xseq); const Kbig=this.Wk.forward(Xseq); const Vbig=this.Wv.forward(Xseq);
    // split heads
    const Qh=this.splitHeads(Qbig), Kh=this.splitHeads(Kbig), Vh=this.splitHeads(Vbig);
    // compute attention per head per sequence (batch over sequences)
    const H=this.nHeads, Hd=this.headDim; const B = this.BT/this.T;
    this.cache={Qh,Kh,Vh}; this.attnWeights=[]; // for backward
    const outHeads=[];
    for(let h=0; h<H; h++){
      // reshape to B x T x Hd
      const Q = this.view3D(Qh[h], B, this.T, Hd);
      const K = this.view3D(Kh[h], B, this.T, Hd);
      const V = this.view3D(Vh[h], B, this.T, Hd);
      const Oh = mat(B*this.T, Hd); // output for this head
      // per batch element compute causal attention
      for(let b=0;b<B;b++){
        // get slices (T x Hd)
        const Qb = this.sliceBT(Qh[h], b, this.T);
        const Kb = this.sliceBT(Kh[h], b, this.T);
        const Vb = this.sliceBT(Vh[h], b, this.T);
        // scores = Q * K^T / sqrt(Hd)  => (T x T)
        const Kbt = transpose(Kb);
        let scores = matmul(Qb, Kbt);
        mulScalarInPlace(scores, 1/Math.sqrt(Hd));
        // causal mask: j>i => -Inf
        for(let i=0;i<this.T;i++) for(let j=0;j<this.T;j++) if(j>i) set(scores,i,j, -1e9);
        // softmax row-wise
        const att = mat(this.T,this.T);
        for(let i=0;i<this.T;i++){
          const row = new Float64Array(this.T);
          for(let j=0;j<this.T;j++) row[j]=get(scores,i,j);
          softmaxInPlace(row);
          for(let j=0;j<this.T;j++) set(att,i,j,row[j]);
        }
        // save attn for backward
        this.attnWeights.push(att);
        // out = att * Vb  => (T x Hd)
        const out = matmul(att, Vb);
        // write back to Oh positions for batch b
        for(let i=0;i<this.T;i++) for(let j=0;j<Hd;j++) set(Oh, b*this.T+i, j, get(out,i,j));
      }
      outHeads.push(Oh);
    }
    const concat = this.concatHeads(outHeads);
    const Y = this.Wo.forward(concat);
    // save for backward
    this.cache.outHeads = outHeads;
    this.cache.concat = concat;
    return Y;
  }
  backward(dY){
    // through Wo
    const dConcat = this.Wo.backward(dY);
    const H=this.nHeads, Hd=this.headDim; const B = this.BT/this.T;
    // split gradient to heads
    const dHeads=[]; for(let h=0;h<H;h++){ dHeads.push(mat(B*this.T, Hd)); }
    for(let i=0;i<dConcat.rows;i++) for(let j=0;j<dConcat.cols;j++){
      const h = Math.floor(j/Hd); const jj = j - h*Hd; set(dHeads[h], i, jj, get(dConcat,i,j));
    }
    const dQbig = mat(this.BT, this.D), dKbig=mat(this.BT,this.D), dVbig=mat(this.BT,this.D);
    // per head backward attention
    let attnIdx=0;
    for(let h=0; h<H; h++){
      const Qh=this.cache.Qh[h], Kh=this.cache.Kh[h], Vh=this.cache.Vh[h];
      const Hd=this.headDim; const dQh=mat(Qh.rows,Qh.cols), dKh=mat(Kh.rows,Kh.cols), dVh=mat(Vh.rows,Vh.cols);
      const B=this.BT/this.T;
      for(let b=0;b<B;b++){
        // slices (T x Hd)
        const Qb = this.sliceBT(Qh, b, this.T);
        const Kb = this.sliceBT(Kh, b, this.T);
        const Vb = this.sliceBT(Vh, b, this.T);
        const dOb = this.sliceBT(dHeads[h], b, this.T);
        const att = this.attnWeights[attnIdx++]; // (T x T)
        // out = att * Vb => dAtt = dOut * Vb^T ; dVb = att^T * dOut
        const Vbt = transpose(Vb);
        const attT = transpose(att);
        const dAtt = matmul(dOb, Vbt);
        const dVb = matmul(attT, dOb);
        // softmax backward: for each row i, dScores_i = J_softmax^T * dAtt_i
        const dScores = mat(this.T, this.T);
        for(let i=0;i<this.T;i++){
          // s = att[i, :]
          const s = new Float64Array(this.T);
          for(let j=0;j<this.T;j++) s[j]=get(att,i,j);
          // Jacobian-vector product efficiently: dScores_j = s_j*(dAtt_j - sum_k s_k*dAtt_k)
          let dot=0; for(let j=0;j<this.T;j++) dot += s[j]*get(dAtt,i,j);
          for(let j=0;j<this.T;j++){
            const ds = s[j]*(get(dAtt,i,j)-dot);
            set(dScores,i,j, ds);
          }
        }
        // scores = (Qb * Kb^T)/sqrt(Hd) (with causal mask)
        // dQb = dScores * Kb / sqrt(Hd)
        // dKb = dScores^T * Qb / sqrt(Hd)
        const Kbt = transpose(Kb);
        const dQb = matmul(dScores, Kb); mulScalarInPlace(dQb, 1/Math.sqrt(Hd));
        const dKb = matmul(transpose(dScores), Qb); mulScalarInPlace(dKb, 1/Math.sqrt(Hd));
        // write back to big dQh/dKh/dVh
        for(let i=0;i<this.T;i++) for(let j=0;j<Hd;j++){
          set(dQh, b*this.T+i, j, get(dQb,i,j));
          set(dKh, b*this.T+i, j, get(dKb,i,j));
          set(dVh, b*this.T+i, j, get(dVb,i,j));
        }
      }
      // accumulate into big Q,K,V grads through the projection splits
      addInPlace(dQbig, dQh); addInPlace(dKbig, dKh); addInPlace(dVbig, dVh);
    }
    // back through Q,K,V projection linears
    const dQproj = this.Wq.backward(dQbig);
    const dKproj = this.Wk.backward(dKbig);
    const dVproj = this.Wv.backward(dVbig);
    // inputs of Q,K,V are the same Xseq; total grad is sum
    const dX = add(add(dQproj, dKproj), dVproj);
    return dX;
  }
  view3D(M, B, T, D){ // returns a logical view; we still slice via sliceBT
    return {B,T,D,M};
  }
  sliceBT(M, b, T){ // (B*T, D) -> (T, D) slice for batch b
    const D=M.cols; const out=mat(T,D);
    for(let i=0;i<T;i++) for(let j=0;j<D;j++) set(out,i,j, get(M, b*T+i, j));
    return out;
  }
  step(opt,t){ this.Wq.step(opt,t); this.Wk.step(opt,t); this.Wv.step(opt,t); this.Wo.step(opt,t); }
  zeroGrad(){ this.Wq.zeroGrad(); this.Wk.zeroGrad(); this.Wv.zeroGrad(); this.Wo.zeroGrad(); }
  params(){ return [
    {name:'WqW',M:this.Wq.W},{name:'WqB',M:this.Wq.b},
    {name:'WkW',M:this.Wk.W},{name:'WkB',M:this.Wk.b},
    {name:'WvW',M:this.Wv.W},{name:'WvB',M:this.Wv.b},
    {name:'WoW',M:this.Wo.W},{name:'WoB',M:this.Wo.b},
  ]; }
}

class TransformerBlock {
  constructor(dModel, nHeads, ffnHidden){
    this.ln1=new LayerNorm(dModel); this.mha=new MultiHeadAttention(dModel,nHeads);
    this.ln2=new LayerNorm(dModel); this.ffn=new FeedForward(dModel, ffnHidden);
  }
  forward(X, T){
    const X1 = this.ln1.forward(X);
    const att = this.mha.forward(X1, T);
    const Y = add(att, X); // residual
    const Y1 = this.ln2.forward(Y);
    const ffn = this.ffn.forward(Y1);
    const Z = add(ffn, Y); // residual
    this.cache={X, X1, att, Y, Y1, ffn, Z, T};
    return Z;
  }
  backward(dZ){
    const {X, X1, att, Y, Y1, ffn, T} = this.cache;
    // residual back: Z = ffn + Y
    const dffn = matCopy(dZ); const dY = matCopy(dZ);
    const dY1 = this.ffn.backward(dffn);
    const dYln = this.ln2.backward(dY1);
    addInPlace(dY, dYln); // accumulate into Y path
    // Y = att + X
    const datt = matCopy(dY); const dX_res = matCopy(dY);
    const dX1 = this.mha.backward(datt);
    const dXln = this.ln1.backward(dX1);
    addInPlace(dX_res, dXln);
    return dX_res;
  }
  step(opt,t){ this.ln1.step(opt,t); this.mha.step(opt,t); this.ln2.step(opt,t); this.ffn.step(opt,t); }
  zeroGrad(){ this.ln1.zeroGrad(); this.mha.zeroGrad(); this.ln2.zeroGrad(); this.ffn.zeroGrad(); }
  params(){ return [
    ...this.ln1.params(), ...this.mha.params(), ...this.ln2.params(), ...this.ffn.params()
  ]; }
}

class Embedding {
  constructor(vocabSize, dModel, maxSeq){
    this.TE = kaimingInit(vocabSize, dModel, WEIGHT_INIT_SCALE);
    this.PE = kaimingInit(maxSeq, dModel, WEIGHT_INIT_SCALE);
    this.gTE = mat(vocabSize, dModel); this.gPE = mat(maxSeq, dModel);
  }
  forward(batchIds){ // batchIds shape: B x T (array of arrays of ints)
    const B = batchIds.length; const T = batchIds[0].length;
    const out = mat(B*T, this.TE.cols);
    this.cache={batchIds};
    for(let b=0;b<B;b++){
      for(let t=0;t<T;t++){
        const tok = batchIds[b][t];
        for(let j=0;j<this.TE.cols;j++){
          const val = get(this.TE,tok,j) + get(this.PE,t,j);
          set(out, b*T+t, j, val);
        }
      }
    }
    return out;
  }
  backward(dX){ // dX is (B*T, D)
    const {batchIds} = this.cache; const B=batchIds.length; const T=batchIds[0].length; const D=this.TE.cols;
    // accumulate grads into gTE (per token row) and gPE (per position row)
    for(let b=0;b<B;b++){
      for(let t=0;t<T;t++){
        const tok=batchIds[b][t];
        for(let j=0;j<D;j++){
          const g = get(dX,b*T+t,j);
          set(this.gTE, tok, j, get(this.gTE,tok,j)+g);
          set(this.gPE, t, j, get(this.gPE,t,j)+g);
        }
      }
    }
  }
  step(opt,t){ opt.adamStepSimple(this.TE, this.gTE, this.PE, this.gPE, t); this.zeroGrad(); }
  zeroGrad(){ fillZero(this.gTE); fillZero(this.gPE); }
  params(){ return [{name:'TE',M:this.TE},{name:'PE',M:this.PE}]; }
}

class LMHead {
  constructor(dModel, vocab){ this.proj=new Linear(dModel, vocab); }
  forward(X){ return this.proj.forward(X); }
  backward(dY){ return this.proj.backward(dY); }
  step(opt,t){ this.proj.step(opt,t); }
  zeroGrad(){ this.proj.zeroGrad(); }
  params(){ return this.proj.params(); }
}

class Adam {
  constructor(lr=LR,b1=BETA1,b2=BETA2,eps=EPS){ this.lr=lr; this.b1=b1; this.b2=b2; this.eps=eps; this.m=new Map(); this.v=new Map(); }
  _key(M){ return M.data; }
  _getMV(M){ const k=this._key(M); if(!this.m.has(k)){ this.m.set(k, zeros(M.data.length)); this.v.set(k, zeros(M.data.length)); } return {m:this.m.get(k), v:this.v.get(k)}; }
  adamStep(W, gW, b, gb, t){ this._adamOne(W, gW, t); this._adamOne(b, gb, t); }
  adamStepSimple(A, gA, B, gB, t){ this._adamOne(A,gA,t); this._adamOne(B,gB,t); }
  _adamOne(param, grad, t){ const {m,v}=this._getMV(param); const lr=this.lr; const b1=this.b1, b2=this.b2, eps=this.eps;
    const mt=m, vt=v, p=param.data, g=grad.data; const n=p.length; const b1t = Math.pow(b1,t), b2t = Math.pow(b2,t);
    for(let i=0;i<n;i++){
      mt[i]=b1*mt[i] + (1-b1)*g[i];
      vt[i]=b2*vt[i] + (1-b2)*g[i]*g[i];
      const mhat = mt[i]/(1-b1t); const vhat = vt[i]/(1-b2t);
      p[i] -= lr * mhat / (Math.sqrt(vhat)+eps);
    }
  }
  save(path){ const obj={ lr:this.lr,b1:this.b1,b2:this.b2,eps:this.eps, m:Array.from(this.m.values(),a=>Array.from(a)), v:Array.from(this.v.values(),a=>Array.from(a)) };
    // cannot reliably map back by key across reloads; instead we just drop state on load unless same objects are reused in one session.
    fs.writeFileSync(path, JSON.stringify(obj));
  }
}

/****************\n|* LLM wrapper  *|
\****************/
class TinyGPT {
  constructor(vocabSize){
    this.embed = new Embedding(vocabSize, D_MODEL, MAX_SEQ_LEN);
    this.blocks = Array.from({length:N_LAYERS}, ()=> new TransformerBlock(D_MODEL, N_HEADS, FFN_HIDDEN));
    this.ln_f = new LayerNorm(D_MODEL);
    this.head = new LMHead(D_MODEL, vocabSize);
    this.opt = new Adam(LR,BETA1,BETA2,EPS);
    this.t=1;
  }
  forward(batchIds){ // batchIds: B x T
    const B = batchIds.length; const T = batchIds[0].length;
    let X = this.embed.forward(batchIds); // (B*T, D)
    for(const blk of this.blocks){ X = blk.forward(X, T); }
    X = this.ln_f.forward(X);
    const logits = this.head.forward(X); // (B*T, V)
    this.cache={T,B};
    return logits;
  }
  backward(dlogits){ // dlogits shape (B*T, V)
    let dX = this.head.backward(dlogits);
    dX = this.ln_f.backward(dX);
    for(let i=this.blocks.length-1;i>=0;i--){ dX = this.blocks[i].backward(dX); }
    this.embed.backward(dX);
  }
  step(){ const t=this.t++;
    for(const blk of this.blocks){ blk.step(this.opt,t); }
    this.ln_f.step(this.opt,t); this.head.step(this.opt,t); this.embed.step(this.opt,t);
  }
  zeroGrad(){ for(const blk of this.blocks) blk.zeroGrad(); this.ln_f.zeroGrad(); this.head.zeroGrad(); this.embed.zeroGrad(); }
  params(){ return [ ...this.embed.params(), ...this.blocks.flatMap(b=>b.params()), ...this.ln_f.params(), ...this.head.params() ]; }
}

/*********************\n|* Loss & Training  *|
\*********************/
function crossEntropyLoss(logits, targets, vocabSize){
  // logits: (B*T, V) ; targets: (B*T) int ids
  const N = logits.rows; const V = logits.cols;
  let loss=0; const probs = mat(N,V);
  // softmax per row + loss
  for(let i=0;i<N;i++){
    const row = new Float64Array(V);
    for(let j=0;j<V;j++) row[j]=get(logits,i,j);
    softmaxInPlace(row);
    const t = targets[i]; const p = Math.max(row[t], 1e-12);
    loss += -Math.log(p);
    for(let j=0;j<V;j++) set(probs,i,j,row[j]);
  }
  loss/=N;
  // dlogits = probs; dlogits[t] -= 1 ; then /N
  const dlogits = mat(N,V);
  for(let i=0;i<N;i++){
    for(let j=0;j<V;j++) set(dlogits,i,j, get(probs,i,j));
    const t = targets[i]; set(dlogits,i,t, get(dlogits,i,t)-1);
  }
  mulScalarInPlace(dlogits, 1/N);
  return {loss, dlogits};
}

function makeBatches(ids, B, T){
  // returns list of batches; each batch is {x: BxT, y: BxT}
  const batches=[]; const total = Math.floor((ids.length-1)/(B*T));
  for(let n=0;n<total;n++){
    const x=[], y=[]; const start=n*B*T;
    for(let b=0;b<B;b++){
      const s = start+b*T; const seqX = ids.slice(s, s+T); const seqY = ids.slice(s+1, s+T+1);
      x.push(seqX); y.push(seqY);
    }
    batches.push({x, y});
  }
  return batches;
}

function flatten2D(arr){ const B=arr.length, T=arr[0].length; const out=new Int32Array(B*T); let k=0; for(let b=0;b<B;b++) for(let t=0;t<T;t++) out[k++]=arr[b][t]; return out; }

function saveWeights(model, tokenizer){
  const obj = { tokenizer:{itos:model.tokenizer.itos}, params:{} };
  for(const p of model.params()){
    obj.params[p.name] = {rows:p.M.rows, cols:p.M.cols, data:Array.from(p.M.data)};
  }
  fs.writeFileSync(CKPT_WEIGHTS, JSON.stringify(obj));
  console.log('Saved weights ->', CKPT_WEIGHTS);
}

function loadWeightsIfAny(model){
  if (!fs.existsSync(CKPT_WEIGHTS)) return false;
  const raw = JSON.parse(fs.readFileSync(CKPT_WEIGHTS,'utf8'));
  const P = new Map(model.params().map(p=>[p.name,p.M]));
  for(const [name, blob] of Object.entries(raw.params)){
    const M = P.get(name); if(!M) continue; if(M.rows!==blob.rows || M.cols!==blob.cols) continue;
    M.data.set(blob.data);
  }
  console.log('Loaded weights from', CKPT_WEIGHTS);
  return true;
}

/****************\n|* Inference    *|
\****************/
function generate(model, prompt, maxNew=20){
  const tok = model.tokenizer; let ids = tok.encode(prompt, true);
  for(let step=0; step<maxNew; step++){
    const ctx = ids.slice(Math.max(0, ids.length-MAX_SEQ_LEN), ids.length);
    // pad if shorter
    const pad = Array.from({length:MAX_SEQ_LEN-ctx.length}, ()=>tok.stoi['<pad>']);
    const x = [pad.concat(ctx)]; // B=1, T=MAX_SEQ_LEN
    const logits = model.forward(x);
    const V = logits.cols; const lastRow = logits.rows-1;
    const row = new Float64Array(V); for(let j=0;j<V;j++) row[j]=get(logits,lastRow,j);
    softmaxInPlace(row);
    // sample greedily (or sample by prob)
    const nextId = argmax(row);
    ids.push(nextId);
    if(nextId === tok.stoi['<eos>']) break;
  }
  // drop initial BOS
  const decoded = tok.decode(ids.slice(1));
  return decoded;
}

/****************\n|* Demo script  *|
\****************/
const corpusText = `
Hello world this is a tiny example corpus for a word level tokenizer.
this corpus is very small but it will let us run a micro training loop.
hello world again this time we test the model by predicting the next word.
`;

(function main(){
  // 1) Tokenizer
  const tokenizer = new WordTokenizer();
  tokenizer.build(corpusText);
  const vocabSize = tokenizer.size();
  console.log('Vocab size =', vocabSize);

  // 2) Prepare data ids (single long stream)
  const ids = tokenizer.encode(corpusText, true);

  // 3) Model
  const model = new TinyGPT(vocabSize);
  model.tokenizer = tokenizer; // attach for saving
  loadWeightsIfAny(model);

  // 4) Training batches
  const batches = makeBatches(ids, BATCH_SIZE, MAX_SEQ_LEN);
  console.log('Total batches:', batches.length);

  // 5) Train a few steps (for demo)
  const STEPS = Math.min(5, batches.length);
  for(let step=0; step<STEPS; step++){
    const {x,y} = batches[step];
    const logits = model.forward(x);
    // targets flatten
    const yFlat = flatten2D(y);
    const {loss, dlogits} = crossEntropyLoss(logits, yFlat, vocabSize);
    model.zeroGrad();
    model.backward(dlogits);
    model.step();
    console.log(`Step ${step+1}/${STEPS} - loss: ${loss.toFixed(4)}`);
  }

  // 6) Save weights
  saveWeights(model, tokenizer);

  // 7) Simple generation
  const out = generate(model, 'hello world', 10);
  console.log('\nGenerated:', out);
})();
