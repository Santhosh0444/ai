// mini_gpt_trainer.js
// Pure JS GPT-style decoder-only transformer with Adam, batching, clipping, and sampling.
// Node.js only (no external libraries)

const fs = require('fs');

// ---------- CONFIG ----------
const CONFIG = {
  d_model: 128,       // smaller default for demo; change to 512 for full size (slow)
  n_heads: 8,         // must divide d_model
  d_ff: 512,          // usually 4 * d_model
  n_layers: 2,        // number of transformer blocks
  max_seq_len: 32,    // context length
  lr: 2e-4,
  betas: [0.9, 0.999],
  eps: 1e-8,
  batch_size: 2,
  epochs: 4,
  grad_clip: 1.0,     // clip grads element-wise to [-grad_clip,grad_clip]
  save_path: './weights_trained.json',
  vocab_min_freq: 1,
  top_k: 40,
  top_p: 0.9
};

// ---------- NUMERIC HELPERS ----------
function zeros(shape) { if (shape.length===1) return Array(shape[0]).fill(0); const [n,...rest]=shape; return Array.from({length:n},()=>zeros(rest)); }
function randn(shape, scale=0.02) {
  function gaussian(){ let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
  if (shape.length===1) return Array(shape[0]).fill(0).map(()=>gaussian()*scale);
  const [n,...rest]=shape; return Array.from({length:n},()=>randn(rest,scale));
}
function matmul(A,B){ // A: m x k, B: k x n
  const m=A.length, k=A[0].length, n=B[0].length;
  const C=Array.from({length:m},()=>Array(n).fill(0));
  for(let i=0;i<m;i++){
    for(let t=0;t<k;t++){
      const v=A[i][t];
      if (v===0) continue;
      const rowB=B[t];
      for(let j=0;j<n;j++) C[i][j]+=v*rowB[j];
    }
  }
  return C;
}
function addBias(X,b){ const m=X.length,n=X[0].length; return Array.from({length:m},(_,i)=>X[i].map((v,j)=>v+b[j])); }
function add(X,Y){ return X.map((r,i)=>r.map((v,j)=>v+Y[i][j])); }
function sub(X,Y){ return X.map((r,i)=>r.map((v,j)=>v-Y[i][j])); }
function mulScalar(X,s){ if(Array.isArray(X[0])) return X.map(row=>row.map(v=>v*s)); return X.map(v=>v*s); }
function transpose(A){ const m=A.length,n=A[0].length; const B=Array.from({length:n},()=>Array(m).fill(0)); for(let i=0;i<m;i++) for(let j=0;j<n;j++) B[j][i]=A[i][j]; return B; }
function softmaxRowStable(row){ const mx=Math.max(...row); const ex=row.map(v=>Math.exp(v-mx)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(v=>v/s); }
function softmaxRows(X){ return X.map(softmaxRowStable); }
function argmax(row){ let idx=0; for(let i=1;i<row.length;i++) if(row[i]>row[idx]) idx=i; return idx; }
function clipValue(v, c){ if (v>c) return c; if (v<-c) return -c; return v; }

// ---------- TOKENIZER (word-level) ----------
class Tokenizer {
  constructor(){
    this.vocab={'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3};
    this.id2tok=['<pad>','<unk>','<bos>','<eos>'];
    this.freq={};
    this.vocab_size=4;
  }
  feedText(text){
    const words=text.split(/\s+/).filter(Boolean);
    for(const w of words) this.freq[w]=(this.freq[w]||0)+1;
  }
  buildVocab(minFreq=1, maxVocab=50000){
    const items=Object.entries(this.freq).filter(([w,f])=>f>=minFreq);
    items.sort((a,b)=>b[1]-a[1]);
    for(const [w,f] of items.slice(0,maxVocab)){
      if (!(w in this.vocab)){
        this.vocab[w]=this.id2tok.length;
        this.id2tok.push(w);
      }
    }
    this.vocab_size=this.id2tok.length;
  }
  encode(text, maxLen=CONFIG.max_seq_len){
    const words=text.split(/\s+/).filter(Boolean);
    const ids=[this.vocab['<bos>']];
    for(const w of words) ids.push(this.vocab[w]!==undefined?this.vocab[w]:this.vocab['<unk>']);
    ids.push(this.vocab['<eos>']);
    if (ids.length>maxLen) return ids.slice(0,maxLen);
    while(ids.length<maxLen) ids.push(this.vocab['<pad>']);
    return ids;
  }
  decode(ids){ return ids.map(i => (this.id2tok[i]||'<unk>')).join(' '); }
}

// ---------- LayerNorm ----------
class LayerNorm {
  constructor(dim, eps=1e-5){ this.dim=dim; this.eps=eps; this.g=Array(dim).fill(1); this.b=Array(dim).fill(0); this.mg=zeros([dim]); this.vg=zeros([dim]); this.mb=zeros([dim]); this.vb=zeros([dim]); }
  forward(batchRows){ // batchRows: N x dim
    this.cache={mean:[],invstd:[],X:batchRows};
    const out = [];
    for(const row of batchRows){
      const mean = row.reduce((a,b)=>a+b,0)/this.dim;
      const diff = row.map(v=>v-mean);
      const varr = diff.reduce((a,b)=>a+b*b,0)/this.dim;
      const invstd = 1/Math.sqrt(varr+this.eps);
      this.cache.mean.push(mean); this.cache.invstd.push(invstd);
      const norm = diff.map(v=>v*invstd);
      out.push(norm.map((v,i)=>v*this.g[i]+this.b[i]));
    }
    return out;
  }
  backward(gradRows){
    const X=this.cache.X, N=gradRows.length, D=this.dim;
    this.dg=Array(D).fill(0); this.db=Array(D).fill(0);
    const dx = Array.from({length:N},()=>Array(D).fill(0));
    for(let i=0;i<N;i++){
      const row=X[i], mean=this.cache.mean[i], invstd=this.cache.invstd[i];
      const xmu = row.map(v=>v-mean);
      const grad = gradRows[i];
      const sum_dy = grad.reduce((a,b)=>a+b,0);
      const sum_dy_xmu = grad.reduce((a,b,j)=>a + b*xmu[j],0);
      for(let j=0;j<D;j++){
        this.db[j]+=grad[j];
        this.dg[j]+=grad[j]* (xmu[j]*invstd);
      }
      for(let j=0;j<D;j++){
        const gij = this.g[j];
        dx[i][j] = (1/D)*invstd * gij * ( D*grad[j] - sum_dy - (xmu[j]*invstd*invstd)*sum_dy_xmu );
      }
    }
    return dx;
  }
}

// ---------- Linear ----------
class Linear {
  constructor(inDim, outDim, name='lin'){
    this.inDim=inDim; this.outDim=outDim; this.name=name;
    this.W = randn([inDim, outDim], Math.sqrt(2/(inDim+outDim)));
    this.b = Array(outDim).fill(0);
    this.dW=zeros([inDim, outDim]); this.db=Array(outDim).fill(0);
    this.mW=zeros([inDim,outDim]); this.vW=zeros([inDim,outDim]);
    this.mb=Array(outDim).fill(0); this.vb=Array(outDim).fill(0);
  }
  forward(X){ this.X=X; const Y = addBias(matmul(X,this.W), this.b); return Y; }
  backward(dY){
    const X=this.X, m=X.length;
    const Xt = transpose(X);
    const dW = matmul(Xt, dY);
    const db = Array(dY[0].length).fill(0);
    for(let i=0;i<dY.length;i++) for(let j=0;j<dY[0].length;j++) db[j]+=dY[i][j];
    const WT = transpose(this.W);
    const dX = matmul(dY, WT);
    // average over batch
    this.dW = dW.map(row=>row.map(v=>v/m));
    this.db = db.map(v=>v/m);
    return dX;
  }
}

// ---------- GELU ----------
function gelu(x){ return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)))); }
function geluRow(row){ return row.map(gelu); }
function geluBackward(pre, gradOut){
  const out = Array(pre.length);
  for(let i=0;i<pre.length;i++){
    const x=pre[i];
    const tanhVal = Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)));
    const dTan = (1 - tanhVal*tanhVal) * Math.sqrt(2/Math.PI) * (1 + 0.134145 * x*x);
    const dx = 0.5*(1+tanhVal) + 0.5*x*dTan;
    out[i] = gradOut[i]*dx;
  }
  return out;
}

// ---------- MultiHeadAttention ----------
class MultiHeadAttention {
  constructor(d_model, n_heads, name='mha'){
    this.d_model=d_model; this.n_heads=n_heads; this.d_head=d_model/n_heads;
    if(!Number.isInteger(this.d_head)) throw new Error('d_model must be divisible by n_heads');
    this.q=new Linear(d_model,d_model,name+'.q'); this.k=new Linear(d_model,d_model,name+'.k');
    this.v=new Linear(d_model,d_model,name+'.v'); this.o=new Linear(d_model,d_model,name+'.o');
    this.name=name;
  }
  splitHeads(batchSeq) { // batch x seq x d_model -> batch x seq x heads x d_head
    const B=batchSeq.length, S=batchSeq[0].length;
    return Array.from({length:B},(_,b)=>Array.from({length:S},(_,i)=>{
      const row = batchSeq[b][i];
      const heads = [];
      for(let h=0;h<this.n_heads;h++) heads.push(row.slice(h*this.d_head,(h+1)*this.d_head));
      return heads;
    }));
  }
  mergeHeads(H){ // batch x seq x heads x d_head -> batch x seq x d_model
    const B=H.length, S=H[0].length;
    return Array.from({length:B},(_,b)=>Array.from({length:S},(_,i)=>H[b][i].flat()));
  }
  forward(X){ // X: batch x seq x d_model
    const B=X.length, S=X[0].length;
    const flat = X.flat();
    const Qflat = this.q.forward(flat); const Kflat = this.k.forward(flat); const Vflat = this.v.forward(flat);
    const Q = Array.from({length:B},(_,b)=>Qflat.slice(b*S,(b+1)*S));
    const K = Array.from({length:B},(_,b)=>Kflat.slice(b*S,(b+1)*S));
    const V = Array.from({length:B},(_,b)=>Vflat.slice(b*S,(b+1)*S));
    const Qh = this.splitHeads(Q); const Kh = this.splitHeads(K); const Vh = this.splitHeads(V);
    this.cache = {X,Qh,Kh,Vh};
    const outHeads = Array.from({length:B},()=>Array.from({length:S},()=>Array.from({length:this.n_heads},()=>Array(this.d_head).fill(0))));
    for(let b=0;b<B;b++){
      for(let h=0;h<this.n_heads;h++){
        const Qmat = Qh[b].map(r=>r[h]); const Kmat = Kh[b].map(r=>r[h]); const Vmat = Vh[b].map(r=>r[h]);
        const scale = 1/Math.sqrt(this.d_head);
        // compute scores seq x seq
        const scores = Array.from({length:S},()=>Array(S).fill(0));
        for(let i=0;i<S;i++) for(let j=0;j<S;j++){
          let s=0;
          const qi = Qmat[i], kj = Kmat[j];
          for(let d=0;d<this.d_head;d++) s += qi[d]*kj[d];
          scores[i][j] = s*scale;
        }
        // causal mask
        for(let i=0;i<S;i++) for(let j=i+1;j<S;j++) scores[i][j] = -1e9;
        const attn = softmaxRows(scores);
        // output = attn * Vmat -> seq x d_head
        for(let i=0;i<S;i++){
          const row = Array(this.d_head).fill(0);
          for(let j=0;j<S;j++){
            const a = attn[i][j];
            for(let d=0;d<this.d_head;d++) row[d]+= a*Vmat[j][d];
          }
          outHeads[b][i][h] = row;
        }
      }
    }
    const merged = this.mergeHeads(outHeads);
    const flatMerged = merged.flat();
    const Yflat = this.o.forward(flatMerged);
    const Y = Array.from({length:B},(_,b)=>Yflat.slice(b*S,(b+1)*S));
    this.cache.attn_out = outHeads;
    return Y;
  }
  backward(dY){
    const B=dY.length, S=dY[0].length;
    const dYflat = dY.flat();
    const dMergedFlat = this.o.backward(dYflat);
    const dMerged = Array.from({length:B},(_,b)=>dMergedFlat.slice(b*S,(b+1)*S));
    // split into heads
    const dHeads = Array.from({length:B},(_,b)=>Array.from({length:S},(_,i)=>{
      const arr = dMerged[b][i]; const heads=[]; for(let h=0;h<this.n_heads;h++) heads.push(arr.slice(h*this.d_head,(h+1)*this.d_head)); return heads;
    }));
    const dQ = Array.from({length:B},()=>Array.from({length:S},()=>Array(this.d_model).fill(0)));
    const dK = Array.from({length:B},()=>Array.from({length:S},()=>Array(this.d_model).fill(0)));
    const dV = Array.from({length:B},()=>Array.from({length:S},()=>Array(this.d_model).fill(0)));
    const Qh=this.cache.Qh, Kh=this.cache.Kh, Vh=this.cache.Vh;
    for(let b=0;b<B;b++){
      for(let h=0;h<this.n_heads;h++){
        const Qmat=Qh[b].map(r=>r[h]); const Kmat=Kh[b].map(r=>r[h]); const Vmat=Vh[b].map(r=>r[h]);
        const dHead = dHeads[b].map(r=>r[h]);
        const scale=1/Math.sqrt(this.d_head);
        // compute scores and attn again
        const scores = Array.from({length:S},()=>Array(S).fill(0));
        for(let i=0;i<S;i++) for(let j=0;j<S;j++){ let s=0; for(let d=0;d<this.d_head;d++) s+=Qmat[i][d]*Kmat[j][d]; scores[i][j]=s*scale; }
        for(let i=0;i<S;i++) for(let j=i+1;j<S;j++) scores[i][j]=-1e9;
        const attn = softmaxRows(scores);
        // dV
        for(let j=0;j<S;j++){
          const accum = Array(this.d_head).fill(0);
          for(let i=0;i<S;i++){ const a=attn[i][j]; for(let d=0;d<this.d_head;d++) accum[d] += a * dHead[i][d]; }
          dV[b][j].splice(h*this.d_head, this.d_head, ...accum);
        }
        // dAttn = dHead dot V
        const dAttn = Array.from({length:S},()=>Array(S).fill(0));
        for(let i=0;i<S;i++) for(let j=0;j<S;j++){
          let s=0; for(let d=0;d<this.d_head;d++) s += dHead[i][d] * Vmat[j][d]; dAttn[i][j]=s;
        }
        // softmax back
        const dScores = Array.from({length:S},()=>Array(S).fill(0));
        for(let i=0;i<S;i++){
          const a=attn[i]; const rowGrad=dAttn[i];
          const ssum = a.reduce((acc,ai,idx)=>acc + ai * rowGrad[idx],0);
          for(let j=0;j<S;j++) dScores[i][j] = a[j] * (rowGrad[j] - ssum);
        }
        // scale
        for(let i=0;i<S;i++) for(let j=0;j<S;j++) dScores[i][j] *= scale;
        // dQ and dK
        for(let i=0;i<S;i++){
          const accum = Array(this.d_head).fill(0);
          for(let j=0;j<S;j++){ const s=dScores[i][j]; for(let d=0;d<this.d_head;d++) accum[d]+= s * Kmat[j][d]; }
          dQ[b][i].splice(h*this.d_head, this.d_head, ...accum);
        }
        for(let j=0;j<S;j++){
          const accum = Array(this.d_head).fill(0);
          for(let i=0;i<S;i++){ const s=dScores[i][j]; for(let d=0;d<this.d_head;d++) accum[d]+= s * Qmat[i][d]; }
          dK[b][j].splice(h*this.d_head, this.d_head, ...accum);
        }
      }
    }
    // flatten and send through q,k,v linear backward
    const dQflat = dQ.flat(), dKflat = dK.flat(), dVflat = dV.flat();
    const dXq = this.q.backward(dQflat);
    const dXk = this.k.backward(dKflat);
    const dXv = this.v.backward(dVflat);
    // sum contributions
    const dXsum = dXq.map((row,i)=>row.map((v,j)=>v + dXk[i][j] + dXv[i][j]));
    const dX = Array.from({length:B},(_,b)=>dXsum.slice(b*S,(b+1)*S));
    return dX;
  }
  getParams(){ return Object.assign({}, this.q.W, this.k.W, this.v.W, this.o.W); } // not used directly here
}

// ---------- FeedForward ----------
class FeedForward {
  constructor(d_model,d_ff,name='ffn'){
    this.lin1=new Linear(d_model,d_ff,name+'.lin1'); this.lin2=new Linear(d_ff,d_model,name+'.lin2'); this.name=name;
  }
  forward(X){
    this.X = X;
    const flat = X.flat();
    this.h1 = this.lin1.forward(flat); // (B*S) x d_ff
    this.g = this.h1.map(geluRow);
    const outFlat = this.lin2.forward(this.g);
    const B=X.length, S=X[0].length;
    return Array.from({length:B},(_,b)=>outFlat.slice(b*S,(b+1)*S));
  }
  backward(dY){
    const flat_dY = dY.flat();
    const dH2 = this.lin2.backward(flat_dY);
    const dG = dH2.map((row,i)=>geluBackward(this.h1[i], row));
    const dXflat = this.lin1.backward(dG);
    const B=dY.length, S=dY[0].length;
    return Array.from({length:B},(_,b)=>dXflat.slice(b*S,(b+1)*S));
  }
}

// ---------- Transformer Block ----------
class TransformerBlock {
  constructor(d_model,n_heads,d_ff,name='block'){
    this.mha = new MultiHeadAttention(d_model,n_heads,name+'.mha');
    this.ln1 = new LayerNorm(d_model);
    this.ffn = new FeedForward(d_model,d_ff,name+'.ffn');
    this.ln2 = new LayerNorm(d_model);
    this.name = name;
  }
  forward(X){ // X: batch x seq x d_model
    this.X = X;
    const mha_out = this.mha.forward(X);
    const res1 = add(X, mha_out);
    // ln1 expects rows: treat each token embedding as a row
    const B=X.length, S=X[0].length;
    const ln1_in = [];
    for(let i=0;i<B;i++) for(let j=0;j<S;j++) ln1_in.push(res1[i][j]);
    const ln1_out_flat = this.ln1.forward(ln1_in);
    const ln1_out = Array.from({length:B},(_,i)=>ln1_out_flat.slice(i*S,(i+1)*S));
    const ffn_out = this.ffn.forward(ln1_out);
    const res2 = add(ln1_out, ffn_out);
    const ln2_in = [];
    for(let i=0;i<B;i++) for(let j=0;j<S;j++) ln2_in.push(res2[i][j]);
    const ln2_out_flat = this.ln2.forward(ln2_in);
    const out = Array.from({length:B},(_,i)=>ln2_out_flat.slice(i*S,(i+1)*S));
    this.cache = {mha_out,res1,ln1_out,ffn_out,res2};
    return out;
  }
  backward(dOut){
    const B=dOut.length, S=dOut[0].length;
    // ln2 backward
    const dOutFlat = dOut.flat();
    const dRes2Flat = this.ln2.backward(dOutFlat);
    const dRes2 = Array.from({length:B},(_,i)=>dRes2Flat.slice(i*S,(i+1)*S));
    const dFfn = dRes2.map((r,i)=>r.map(v=>v));
    const dLn1_from_res2 = dRes2.map((r,i)=>r.map(v=>v));
    // ffn backward
    const dLn1_from_ffn = this.ffn.backward(dFfn);
    // total ln1 grad
    const dLn1_total = dLn1_from_res2.map((row,i)=>row.map((v,j)=>v + dLn1_from_ffn[i][j]));
    // ln1 backward
    const dLn1Flat = dLn1_total.flat();
    const dRes1Flat = this.ln1.backward(dLn1Flat);
    const dRes1 = Array.from({length:B},(_,i)=>dRes1Flat.slice(i*S,(i+1)*S));
    // res1 splits to X and mha_out
    const dX_from_res1 = dRes1.map(row=>row.map(v=>v));
    const dMha_in = dRes1.map(row=>row.map(v=>v));
    // mha backward
    const dX_from_mha = this.mha.backward(dMha_in);
    // total dX
    const dX = dX_from_res1.map((row,i)=>row.map((v,j)=>v + dX_from_mha[i][j]));
    return dX;
  }
}

// ---------- GPT Model ----------
class GPT {
  constructor(tokenizer, config){
    this.tok = tokenizer;
    this.d_model=config.d_model; this.n_heads=config.n_heads; this.d_ff=config.d_ff; this.n_layers=config.n_layers;
    this.max_seq_len=config.max_seq_len; this.vocab_size=tokenizer.vocab_size;
    if (this.d_model % this.n_heads !==0) throw new Error('d_model must be divisible by n_heads');
    this.wte = randn([this.vocab_size, this.d_model], 0.02);
    this.wpe = this.makePositionalEncoding(this.max_seq_len, this.d_model);
    this.blocks = [];
    for(let i=0;i<this.n_layers;i++) this.blocks.push(new TransformerBlock(this.d_model,this.n_heads,this.d_ff,'block'+i));
    this.ln_f = new LayerNorm(this.d_model);
    // Adam states for embeddings and positional
    this.m_wte = zeros([this.vocab_size,this.d_model]); this.v_wte = zeros([this.vocab_size,this.d_model]);
    this.m_wpe = zeros([this.max_seq_len,this.d_model]); this.v_wpe=zeros([this.max_seq_len,this.d_model]);
    this.t = 0;
  }
  makePositionalEncoding(maxLen, d_model){
    const pe = Array.from({length:maxLen},(_,pos)=>{
      const row = Array(d_model).fill(0);
      for(let i=0;i<d_model;i++){
        const denom = Math.pow(10000, (2*Math.floor(i/2))/d_model);
        if (i%2===0) row[i] = Math.sin(pos/denom);
        else row[i] = Math.cos(pos/denom);
      }
      return row;
    });
    return pe;
  }
  embedInput(ids){ // ids: B x S
    const B=ids.length, S=ids[0].length;
    const out = Array.from({length:B},(_,i)=>Array.from({length:S},(_,j)=>{
      const id = ids[i][j]; const emb = this.wte[id] || this.wte[this.tok.vocab['<unk>']];
      const pos = this.wpe[j] || Array(this.d_model).fill(0);
      return emb.map((v,k)=>v + pos[k]);
    }));
    return out;
  }
  forward(ids){ // ids: B x S
    this.ids = ids;
    let h = this.embedInput(ids);
    for(const blk of this.blocks) h = blk.forward(h);
    const flat = h.flat();
    const ln_out = this.ln_f.forward(flat);
    this.cache = {h,ln_out,ids};
    const out = Array.from({length:ids.length},(_,i)=>ln_out.slice(i*ids[0].length,(i+1)*ids[0].length));
    const flat_out = out.flat();
    const logitsFlat = matmul(flat_out.map(r=>r), transpose(this.wte)); // (B*S) x V
    const logits = Array.from({length:ids.length},(_,i)=>logitsFlat.slice(i*ids[0].length,(i+1)*ids[0].length));
    return logits; // B x S x V
  }
  computeLossAndGrad(logits, labels){
    const B=logits.length, S=logits[0].length;
    const flatLogits = logits.flat();
    const dFlat=[];
    let loss=0;
    for(let i=0;i<flatLogits.length;i++){
      const row = flatLogits[i];
      const p = softmaxRowStable(row);
      const y = labels.flat()[i];
      loss += -Math.log(Math.max(p[y],1e-12));
      const grad = p.map((v,idx)=> (idx===y ? v-1 : v) );
      dFlat.push(grad);
    }
    loss /= (B*S);
    const scale = 1/(B*S);
    const dLogits = Array.from({length:B},(_,bi)=>Array.from({length:S},(_,si)=>dFlat[bi*S+si].map(v=>v*scale)));
    return {loss, dLogits};
  }
  backward(dLogits){
    const B=dLogits.length, S=dLogits[0].length;
    const flat_dLogits = dLogits.flat();
    // gradient wrt flat_out = dLogits * wte
    const dOutFlat = matmul(flat_dLogits, this.wte); // (B*S) x d_model
    // ln_f backward
    const dLnFlat = this.ln_f.backward(dOutFlat);
    const dLn = Array.from({length:B},(_,i)=>dLnFlat.slice(i*S,(i+1)*S));
    // backward through blocks
    let grad = dLn;
    for(let i=this.blocks.length-1;i>=0;i--) grad = this.blocks[i].backward(grad);
    // grad is w.r.t input embeddings (x = wte[id] + wpe[pos])
    // accumulate dWTE and dWPE
    this.dWTE = zeros([this.vocab_size, this.d_model]);
    this.dWPE = zeros([this.max_seq_len, this.d_model]);
    for(let i=0;i<B;i++) for(let j=0;j<S;j++){
      const id = this.ids[i][j], g = grad[i][j];
      for(let k=0;k<this.d_model;k++){ this.dWTE[id][k] += g[k]; this.dWPE[j][k] += g[k]; }
    }
    // average over batch
    const scale = 1/B;
    for(let i=0;i<this.vocab_size;i++) for(let k=0;k<this.d_model;k++) this.dWTE[i][k] *= scale;
    for(let i=0;i<this.max_seq_len;i++) for(let k=0;k<this.d_model;k++) this.dWPE[i][k] *= scale;
  }

  // ---------- Adam + gradient clipping + parameter updates ----------
  stepAdam(lr, betas, eps, grad_clip){
    this.t += 1;
    const [b1,b2] = betas;
    // clip embedding grads element-wise
    for(let i=0;i<this.vocab_size;i++) for(let k=0;k<this.d_model;k++) this.dWTE[i][k] = clipValue(this.dWTE[i][k], grad_clip);
    for(let i=0;i<this.max_seq_len;i++) for(let k=0;k<this.d_model;k++) this.dWPE[i][k] = clipValue(this.dWPE[i][k], grad_clip);
    // update embeddings
    for(let i=0;i<this.vocab_size;i++){
      for(let k=0;k<this.d_model;k++){
        const g = this.dWTE[i][k];
        this.m_wte[i][k] = b1*this.m_wte[i][k] + (1-b1)*g;
        this.v_wte[i][k] = b2*this.v_wte[i][k] + (1-b2)*g*g;
        const mhat = this.m_wte[i][k] / (1 - Math.pow(b1,this.t));
        const vhat = this.v_wte[i][k] / (1 - Math.pow(b2,this.t));
        this.wte[i][k] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
    }
    // positional enc
    for(let i=0;i<this.max_seq_len;i++) for(let k=0;k<this.d_model;k++){
      let g = this.dWPE[i][k]; g = clipValue(g, grad_clip);
      this.m_wpe[i][k] = b1*this.m_wpe[i][k] + (1-b1)*g;
      this.v_wpe[i][k] = b2*this.v_wpe[i][k] + (1-b2)*g*g;
      const mhat = this.m_wpe[i][k] / (1 - Math.pow(b1,this.t));
      const vhat = this.v_wpe[i][k] / (1 - Math.pow(b2,this.t));
      this.wpe[i][k] -= lr * mhat / (Math.sqrt(vhat)+eps);
    }
    // update every module's parameters: linears and layernorms inside blocks and final ln_f
    // helper functions
    const updateLinear = (L) => {
      // clip grads elementwise
      for(let i=0;i<L.dW.length;i++) for(let j=0;j<L.dW[0].length;j++) L.dW[i][j] = clipValue(L.dW[i][j], grad_clip);
      for(let j=0;j<L.db.length;j++) L.db[j] = clipValue(L.db[j], grad_clip);
      for(let i=0;i<L.W.length;i++) for(let j=0;j<L.W[0].length;j++){
        L.mW[i][j] = b1*L.mW[i][j] + (1-b1)*L.dW[i][j];
        L.vW[i][j] = b2*L.vW[i][j] + (1-b2)*L.dW[i][j]*L.dW[i][j];
        const mhat = L.mW[i][j] / (1 - Math.pow(b1,this.t));
        const vhat = L.vW[i][j] / (1 - Math.pow(b2,this.t));
        L.W[i][j] -= lr * mhat / (Math.sqrt(vhat)+eps);
      }
      for(let j=0;j<L.b.length;j++){
        L.mb[j] = b1*L.mb[j] + (1-b1)*L.db[j];
        L.vb[j] = b2*L.vb[j] + (1-b2)*L.db[j]*L.db[j];
        const mhat = L.mb[j] / (1 - Math.pow(b1,this.t));
        const vhat = L.vb[j] / (1 - Math.pow(b2,this.t));
        L.b[j] -= lr * mhat / (Math.sqrt(vhat)+eps);
      }
    };
    const updateLayerNorm = (LN) => {
      for(let i=0;i<LN.g.length;i++){
        LN.dg[i] = clipValue(LN.dg[i] || 0, grad_clip);
        LN.db[i] = clipValue(LN.db[i] || 0, grad_clip);
        LN.mg[i] = b1*LN.mg[i] + (1-b1)*LN.dg[i];
        LN.vg[i] = b2*LN.vg[i] + (1-b2)*LN.dg[i]*LN.dg[i];
        const mg_hat = LN.mg[i]/(1-Math.pow(b1,this.t));
        const vg_hat = LN.vg[i]/(1-Math.pow(b2,this.t));
        LN.g[i] -= lr * mg_hat / (Math.sqrt(vg_hat)+eps);
        LN.mb[i] = b1*LN.mb[i] + (1-b1)*LN.db[i];
        LN.vb[i] = b2*LN.vb[i] + (1-b2)*LN.db[i]*LN.db[i];
        const mb_hat = LN.mb[i]/(1-Math.pow(b1,this.t));
        const vb_hat = LN.vb[i]/(1-Math.pow(b2,this.t));
        LN.b[i] -= lr * mb_hat / (Math.sqrt(vb_hat)+eps);
      }
    };
    // walk blocks
    for(const blk of this.blocks){
      // mha linears
      updateLinear(blk.mha.q); updateLinear(blk.mha.k); updateLinear(blk.mha.v); updateLinear(blk.mha.o);
      updateLayerNorm(blk.ln1); updateLayerNorm(blk.ln2);
      updateLinear(blk.ffn.lin1); updateLinear(blk.ffn.lin2);
    }
    // final ln_f
    updateLayerNorm(this.ln_f);
  }

  save(path){
    const data = {config:CONFIG, vocab:this.tok.id2tok, wte:this.wte, wpe:this.wpe};
    for(let i=0;i<this.blocks.length;i++){
      const bname='block'+i, blk=this.blocks[i];
      ['q','k','v','o'].forEach(k=>{
        data[`${bname}.${k}.W`] = blk.mha[k].W;
        data[`${bname}.${k}.b`] = blk.mha[k].b;
      });
      data[`${bname}.ffn.lin1.W`] = blk.ffn.lin1.W; data[`${bname}.ffn.lin1.b`] = blk.ffn.lin1.b;
      data[`${bname}.ffn.lin2.W`] = blk.ffn.lin2.W; data[`${bname}.ffn.lin2.b`] = blk.ffn.lin2.b;
      data[`${bname}.ln1.g`] = blk.ln1.g; data[`${bname}.ln1.b`] = blk.ln1.b;
      data[`${bname}.ln2.g`] = blk.ln2.g; data[`${bname}.ln2.b`] = blk.ln2.b;
    }
    data['ln_f.g']=this.ln_f.g; data['ln_f.b']=this.ln_f.b;
    fs.writeFileSync(path, JSON.stringify(data));
    console.log('Saved model to', path);
  }
  load(path){
    const raw = fs.readFileSync(path,'utf8'); const data = JSON.parse(raw);
    if (data.wte) this.wte = data.wte; if (data.wpe) this.wpe = data.wpe;
    for(let i=0;i<this.blocks.length;i++){
      const bname='block'+i; const blk=this.blocks[i];
      ['q','k','v','o'].forEach(k=>{
        const W=data[`${bname}.${k}.W`], b=data[`${bname}.${k}.b`];
        if (W) blk.mha[k].W = W; if (b) blk.mha[k].b=b;
      });
      const l1W=data[`${bname}.ffn.lin1.W`], l1b=data[`${bname}.ffn.lin1.b`];
      const l2W=data[`${bname}.ffn.lin2.W`], l2b=data[`${bname}.ffn.lin2.b`];
      if (l1W) blk.ffn.lin1.W=l1W; if (l1b) blk.ffn.lin1.b=l1b;
      if (l2W) blk.ffn.lin2.W=l2W; if (l2b) blk.ffn.lin2.b=l2b;
      const ln1g=data[`${bname}.ln1.g`], ln1bb=data[`${bname}.ln1.b`];
      const ln2g=data[`${bname}.ln2.g`], ln2bb=data[`${bname}.ln2.b`];
      if (ln1g) blk.ln1.g=ln1g; if (ln1bb) blk.ln1.b=ln1bb;
      if (ln2g) blk.ln2.g=ln2g; if (ln2bb) blk.ln2.b=ln2bb;
    }
    if (data['ln_f.g']) this.ln_f.g = data['ln_f.g'];
    if (data['ln_f.b']) this.ln_f.b = data['ln_f.b'];
    console.log('Loaded model from', path);
  }

  // ---------- Sampling: temperature, top-k, top-p ----------
  sample(prefix_ids, max_new_tokens=20, temperature=1.0, top_k=40, top_p=0.9){
    let ids = prefix_ids.slice();
    for(let t=0;t<max_new_tokens;t++){
      const context = ids.slice(-this.max_seq_len);
      while(context.length < this.max_seq_len) context.unshift(this.tok.vocab['<pad>']);
      const logits = this.forward([context]); // [1][seq][vocab]
      const lastLogits = logits[0][context.length-1].slice(); // copy
      // temperature
      for(let i=0;i<lastLogits.length;i++) lastLogits[i] = lastLogits[i] / temperature;
      // convert to probabilities safely
      const probs = softmaxRowStable(lastLogits);
      // top-k: zero out everything except top_k
      let indices = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]);
      if (top_k > 0) indices = indices.slice(0, top_k);
      // top-p: keep smallest set where cumulative prob >= top_p
      if (top_p < 1.0){
        let cum=0; const kept=[];
        for(const [p,i] of indices){
          cum += p;
          kept.push([p,i]);
          if (cum >= top_p) break;
        }
        indices = kept;
      }
      // renormalize and sample
      const ps = indices.map(x=>x[0]);
      const sum = ps.reduce((a,b)=>a+b,0) || 1;
      const normalized = ps.map(v=>v/sum);
      let r=Math.random(), cum2=0, pick=indices[0][1];
      for(let i=0;i<normalized.length;i++){ cum2 += normalized[i]; if (r < cum2){ pick = indices[i][1]; break; } }
      ids.push(pick);
      if (pick === this.tok.vocab['<eos>']) break;
    }
    return ids;
  }
}

// ---------- Trainer and batching ----------
function shuffleArray(a){ for(let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } }
function createBatches(examples, batchSize){
  const batches = [];
  for(let i=0;i<examples.length;i+=batchSize) batches.push(examples.slice(i,i+batchSize));
  return batches;
}

async function trainDemo(){
  // Toy data (extend with your dataset)
  const texts = [
    "hello world this is a test",
    "hello there how are you",
    "this is another example text",
    "the model will learn to predict words",
    "this is a small dataset for demo"
  ];
  // tokenizer
  const tok = new Tokenizer();
  for(const t of texts) tok.feedText(t);
  tok.buildVocab(CONFIG.vocab_min_freq, 20000);
  console.log('Vocab size:', tok.vocab_size);
  // model config
  const cfg = {
    d_model: CONFIG.d_model, n_heads: CONFIG.n_heads, d_ff: CONFIG.d_ff,
    n_layers: CONFIG.n_layers, max_seq_len: CONFIG.max_seq_len
  };
  const model = new GPT(tok, cfg);
  // prepare examples
  const examples = texts.map(t => tok.encode(t, CONFIG.max_seq_len));
  // training loop
  for(let ep=0; ep<CONFIG.epochs; ep++){
    shuffleArray(examples);
    const batches = createBatches(examples, CONFIG.batch_size);
    let epochLoss=0;
    for(const batch of batches){
      // batch is array of ids arrays
      const idsBatch = batch.map(x=>x.slice());
      const logits = model.forward(idsBatch);
      const {loss, dLogits} = model.computeLossAndGrad(logits, idsBatch);
      epochLoss += loss;
      model.backward(dLogits);
      model.stepAdam(CONFIG.lr, CONFIG.betas, CONFIG.eps, CONFIG.grad_clip);
    }
    console.log(`Epoch ${ep+1}/${CONFIG.epochs}  avg_loss=${(epochLoss/batches.length).toFixed(6)}`);
    model.save(CONFIG.save_path);
  }
  // sampling demo
  const prompt = "hello";
  const prefix = tok.encode(prompt, CONFIG.max_seq_len).slice(0, CONFIG.max_seq_len);
  const gen = model.sample(prefix, 20, 1.0, CONFIG.top_k, CONFIG.top_p);
  console.log('Generated tokens:', gen);
  console.log('Generated text:', tok.decode(gen));
}

// Run demo when executed
if (require.main === module){
  trainDemo().catch(err=>console.error(err));
}
