// mini_gpt.js
// Pure JavaScript GPT-style decoder-only Transformer with training & save/load
// WARNING: CPU-only, will be slow at large sizes. For experimentation reduce sizes.

// ---------- CONFIG ----------
const fs = require('fs');

const CONFIG = {
  d_model: 512,       // embedding size
  n_heads: 8,         // number of attention heads
  d_ff: 2048,         // feedforward size (4xd_model)
  n_layers: 2,        // number of transformer blocks (set low for quick tests)
  max_seq_len: 64,    // context length
  lr: 1e-4,
  betas: [0.9, 0.999],
  eps: 1e-8,
  vocab_min_freq: 1,  // min freq to include token
  save_path: './weights.json'
};

// ---------- UTILS: small numeric helpers ----------
function zeros(shape) {
  if (shape.length === 1) return Array(shape[0]).fill(0);
  const [n, ...rest] = shape;
  return Array.from({length: n}, () => zeros(rest));
}
function randn(shape, scale=0.02) {
  function gaussian() {
    let u=0,v=0;
    while(u===0) u=Math.random();
    while(v===0) v=Math.random();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
  }
  if (shape.length===1) return Array(shape[0]).fill(0).map(()=>gaussian()*scale);
  const [n,...rest]=shape;
  return Array.from({length:n},()=>randn(rest,scale));
}
function cloneArray(a) { return JSON.parse(JSON.stringify(a)); }
function matmul(A,B) { // A: m x k, B: k x n -> m x n
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
function addBias(X, b) { // X: m x n, b: n
  const m=X.length, n=X[0].length;
  const Y=Array.from({length:m},(_,i)=>X[i].map((v,j)=>v + b[j]));
  return Y;
}
function add(X,Y) { // elementwise add of matrices
  const m=X.length, n=X[0].length;
  const Z=Array.from({length:m},(_,i)=>X[i].map((v,j)=>v + Y[i][j]));
  return Z;
}
function sub(X,Y){const m=X.length,n=X[0].length;return Array.from({length:m},(_,i)=>X[i].map((v,j)=>v-Y[i][j]));}
function mulScalar(X,s){if(Array.isArray(X[0])) return X.map(row=>row.map(v=>v*s)); return X.map(v=>v*s);}
function transpose(A){const m=A.length,n=A[0].length; const B=Array.from({length:n},()=>Array(m).fill(0)); for(let i=0;i<m;i++) for(let j=0;j<n;j++) B[j][i]=A[i][j]; return B;}
function softmaxRows(X){ return X.map(row => {
  const mx = Math.max(...row);
  const ex = row.map(v=>Math.exp(v-mx));
  const s = ex.reduce((a,b)=>a+b,0);
  return ex.map(v=>v/s);
});}
function argmaxRow(row){ let idx=0; for(let i=1;i<row.length;i++) if(row[i]>row[idx]) idx=i; return idx; }

// ---------- TOKENIZER (word-level) ----------
class Tokenizer {
  constructor() {
    this.vocab = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3};
    this.id2tok = ['<pad>','<unk>','<bos>','<eos>'];
    this.freq = {};
  }
  feedText(text) {
    const words = text.split(/\s+/).filter(Boolean);
    for (const w of words) { this.freq[w]=(this.freq[w]||0)+1; }
  }
  buildVocab(minFreq=1, maxVocab=50000) {
    const items = Object.entries(this.freq).filter(([w,f])=>f>=minFreq);
    items.sort((a,b)=>b[1]-a[1]);
    for (const [w,f] of items.slice(0,maxVocab)) {
      if (!(w in this.vocab)) {
        const id = this.id2tok.length;
        this.vocab[w]=id; this.id2tok.push(w);
      }
    }
    this.vocab_size = this.id2tok.length;
  }
  encode(text, maxLen=CONFIG.max_seq_len) {
    const words = text.split(/\s+/).filter(Boolean);
    const ids = [this.vocab['<bos>']];
    for (const w of words) ids.push(this.vocab[w]!==undefined?this.vocab[w]:this.vocab['<unk>']);
    ids.push(this.vocab['<eos>']);
    // pad/truncate
    if (ids.length > maxLen) return ids.slice(0, maxLen);
    while (ids.length < maxLen) ids.push(this.vocab['<pad>']);
    return ids;
  }
  decode(ids) {
    return ids.map(i => (this.id2tok[i]||'<unk>')).join(' ');
  }
}

// ---------- LayerNorm ----------
class LayerNorm {
  constructor(dim, eps=1e-5) { this.dim = dim; this.eps=eps; this.g = Array(dim).fill(1); this.b = Array(dim).fill(0); 
    // Adam states
    this.mg=zeros([dim]); this.vg=zeros([dim]); this.mb=zeros([dim]); this.vb=zeros([dim]); 
  }
  forward(X) { // X: batch x dim
    const m=X.length,d=this.dim;
    this.cache={mean:[],invstd:[],X};
    const out = Array.from({length:m},(_,i)=>{
      const row=X[i];
      const mean = row.reduce((a,b)=>a+b,0)/d;
      const diff = row.map(v=>v-mean);
      const varr = diff.reduce((a,b)=>a+b*b,0)/d;
      const invstd = 1/Math.sqrt(varr+this.eps);
      this.cache.mean.push(mean); this.cache.invstd.push(invstd);
      const norm = diff.map(v=>v*invstd);
      return norm.map((v,j)=>v*this.g[j]+this.b[j]);
    });
    return out;
  }
  backward(gradOutput) { // gradOutput: batch x dim -> grads for g,b and grad input
    const X = this.cache.X, m=X.length, d=this.dim;
    const dx = Array.from({length:m},()=>Array(d).fill(0));
    this.dg = Array(d).fill(0); this.db = Array(d).fill(0);
    for(let i=0;i<m;i++){
      const row = X[i];
      const mean = this.cache.mean[i];
      const invstd = this.cache.invstd[i];
      const xmu = row.map(v=>v-mean);
      const grad = gradOutput[i];
      for(let j=0;j<d;j++){
        this.db[j]+=grad[j];
        this.dg[j]+=grad[j]* (xmu[j]*invstd);
      }
      // propagate to input
      // formula: dx = (1/N) * invstd * g * (N*dy - sum(dy) - xmu*invstd^2 * sum(dy*xmu))
      const sum_dy = grad.reduce((a,b)=>a+b,0);
      const sum_dy_xmu = grad.reduce((a,b,idx)=>a + b * xmu[idx], 0);
      for(let j=0;j<d;j++){
        const gij = this.g[j];
        dx[i][j] = (1/d)*invstd * gij * ( d*grad[j] - sum_dy - (xmu[j]*invstd*invstd)*sum_dy_xmu );
      }
    }
    return dx;
  }
  getParams() { return {g:this.g, b:this.b}; }
  getGrads() { return {dg:this.dg, db:this.db}; }
  getNamedParams(prefix) {
    return {
      [prefix+'.g']: this.g,
      [prefix+'.b']: this.b
    };
  }
}

// ---------- Linear layer ----------
class Linear {
  constructor(inDim, outDim, name='') {
    this.inDim=inDim; this.outDim=outDim; this.name=name;
    this.W = randn([inDim, outDim], Math.sqrt(2/(inDim+outDim)));
    this.b = Array(outDim).fill(0);
    // grads
    this.dW = zeros([inDim,outDim]); this.db = Array(outDim).fill(0);
    // Adam states
    this.mW = zeros([inDim,outDim]); this.vW = zeros([inDim,outDim]);
    this.mb = Array(outDim).fill(0); this.vb = Array(outDim).fill(0);
  }
  forward(X) { // X: batch x inDim -> output: batch x outDim
    this.X = X;
    const Y = addBias(matmul(X, this.W), this.b);
    return Y;
  }
  backward(dY) { // dY: batch x outDim
    const X = this.X;
    const m=X.length;
    // dW = X^T * dY
    const Xt = transpose(X);
    const dW = matmul(Xt, dY); // inDim x outDim
    const db = Array(dY[0].length).fill(0);
    for(let i=0;i<dY.length;i++){
      for(let j=0;j<dY[0].length;j++) db[j]+=dY[i][j];
    }
    // dX = dY * W^T
    const WT = transpose(this.W);
    const dX = matmul(dY, WT);
    // store grads averaged
    this.dW = dW.map(row=>row.map(v=>v / m));
    this.db = db.map(v=>v / m);
    return dX;
  }
  getParamsObj(prefix) {
    return { [prefix+'.W']:this.W, [prefix+'.b']:this.b };
  }
  getGradsObj() {
    return { dW:this.dW, db:this.db };
  }
}

// ---------- GELU ----------
function gelu(x){ return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)))); }
function geluRow(row){ return row.map(gelu); }
function geluBackward(row, gradRow){ // row: pre-activation, gradRow: dLoss/dOutput
  // approximate derivative numerically per element using derivative of GELU formula:
  const out = Array(row.length);
  for(let i=0;i<row.length;i++){
    const x=row[i];
    const tanhVal = Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)));
    const left = 0.5*(1+tanhVal);
    const dTan = (1 - tanhVal*tanhVal) * Math.sqrt(2/Math.PI) * (1 + 0.134145 * x*x);
    const dx = 0.5 * (1 + tanhVal) + 0.5 * x * dTan;
    out[i]=gradRow[i]*dx;
  }
  return out;
}

// ---------- Scaled Dot-Product Attention (with causal mask) and MHA ----------
class MultiHeadAttention {
  constructor(d_model, n_heads, name='mha') {
    this.d_model=d_model; this.n_heads=n_heads;
    this.d_head = d_model / n_heads;
    if (!Number.isInteger(this.d_head)) throw new Error('d_model must be divisible by n_heads');
    // linear projections
    this.q = new Linear(d_model, d_model, name+'.q');
    this.k = new Linear(d_model, d_model, name+'.k');
    this.v = new Linear(d_model, d_model, name+'.v');
    this.o = new Linear(d_model, d_model, name+'.o');
    this.name=name;
  }
  splitHeads(X) { // X: batch x seq x d_model -> returns batch*seq x n_heads x d_head? We'll reshape for simplicity as arrays of length batch: each is seq x n_heads x d_head
    // Represent as [batch][seq][head][d_head]
    const batch = X.length, seq=X[0].length;
    const out = Array.from({length:batch},(_,b)=>Array.from({length:seq},(_,i)=>{
      const row = X[b][i];
      const heads = [];
      for(let h=0;h<this.n_heads;h++){
        heads.push(row.slice(h*this.d_head, (h+1)*this.d_head));
      }
      return heads;
    }));
    return out;
  }
  mergeHeads(H) { // H: batch x seq x heads x d_head -> batch x seq x d_model
    const batch = H.length, seq=H[0].length;
    return Array.from({length:batch},(_,b)=>Array.from({length:seq},(_,i)=>{
      return H[b][i].flat();
    }));
  }
  forward(X) {
    // X: batch x seq x d_model
    const batch=X.length, seq=X[0].length;
    // flatten to batch*seq x d_model for linear layers
    const flat = X.flat();
    // q,k,v as arrays of rows
    const Qflat = this.q.forward(flat); // (batch*seq) x d_model
    const Kflat = this.k.forward(flat);
    const Vflat = this.v.forward(flat);
    // reshape back to batch x seq x d_model
    const Q = Array.from({length:batch},(_,b)=>Qflat.slice(b*seq,(b+1)*seq));
    const K = Array.from({length:batch},(_,b)=>Kflat.slice(b*seq,(b+1)*seq));
    const V = Array.from({length:batch},(_,b)=>Vflat.slice(b*seq,(b+1)*seq));
    // split heads
    const Qh = this.splitHeads(Q);
    const Kh = this.splitHeads(K);
    const Vh = this.splitHeads(V);
    // compute attention per head
    // for each batch, head -> seq x seq attn
    this.cache = {X, Qh, Kh, Vh};
    const outHeads = Array.from({length:batch},()=>Array.from({length:seq},()=>Array.from({length:this.n_heads},()=>Array(this.d_head).fill(0))));
    for(let b=0;b<batch;b++){
      for(let h=0;h<this.n_heads;h++){
        // build Qmat (seq x d_head) and Kmat (seq x d_head) -> compute Q*K^T => seq x seq
        const Qmat = Qh[b].map(r=>r[h]); // seq x d_head
        const Kmat = Kh[b].map(r=>r[h]);
        const Vmat = Vh[b].map(r=>r[h]);
        const KT = transpose(Kmat); // d_head x seq
        const scale = 1/Math.sqrt(this.d_head);
        // compute scores
        const scores = Array.from({length:seq},(_,i)=>Array(seq).fill(0));
        for(let i=0;i<seq;i++){
          for(let j=0;j<seq;j++){
            let s=0;
            const qi = Qmat[i], kj = Kmat[j];
            for(let d=0;d<this.d_head;d++) s+=qi[d]*kj[d];
            scores[i][j]=s*scale;
          }
        }
        // apply causal mask: disallow j>i
        for(let i=0;i<seq;i++) for(let j=i+1;j<seq;j++) scores[i][j] = -1e9;
        // softmax rows
        const attn = softmaxRows(scores); // seq x seq
        // multiply attn * Vmat => seq x d_head
        for(let i=0;i<seq;i++){
          for(let d=0;d<this.d_head;d++){
            let val=0;
            for(let j=0;j<seq;j++) val += attn[i][j] * Vmat[j][d];
            outHeads[b][i][h][d]=val; // but we will fix structure below
          }
        }
        // The above structure is awkward; simpler: set outHeads[b][i][h] as array
        for(let i=0;i<seq;i++){
          const row = Array(this.d_head).fill(0);
          for(let d=0;d<this.d_head;d++){
            let val=0;
            for(let j=0;j<seq;j++) val += attn[i][j] * Vmat[j][d];
            row[d]=val;
          }
          outHeads[b][i][h]=row;
        }
      }
    }
    // merge heads: batch x seq x d_model
    const merged = this.mergeHeads(outHeads);
    // final linear
    const flatMerged = merged.flat();
    const Yflat = this.o.forward(flatMerged); // (batch*seq) x d_model
    const Y = Array.from({length:batch},(_,b)=>Yflat.slice(b*seq,(b+1)*seq));
    this.cache.attn_outputs = outHeads;
    return Y;
  }
  backward(dY) {
    // dY: batch x seq x d_model
    const batch=dY.length, seq=dY[0].length;
    const flat_dY = dY.flat();
    // back through o
    const dMergedFlat = this.o.backward(flat_dY); // returns (batch*seq) x d_model
    const dMerged = Array.from({length:batch},(_,b)=>dMergedFlat.slice(b*seq,(b+1)*seq));
    // split heads from merged grads
    const dHeads = Array.from({length:batch},(_,b)=>Array.from({length:seq},(_,i)=>{
      const arr = dMerged[b][i];
      const heads = [];
      for(let h=0;h<this.n_heads;h++) heads.push(arr.slice(h*this.d_head, (h+1)*this.d_head));
      return heads;
    }));
    // propagate through attention to get dV,dK,dQ then to linear layers v,k,q
    // We'll compute gradient contributions to V,K,Q as flattened (batch*seq) x d_model to feed into linear.backward
    const dQ = Array.from({length:batch},()=>Array.from({length:seq},()=>Array(this.d_model).fill(0)));
    const dK = Array.from({length:batch},()=>Array.from({length:seq},()=>Array(this.d_model).fill(0)));
    const dV = Array.from({length:batch},()=>Array.from({length:seq},()=>Array(this.d_model).fill(0)));
    // For simplicity and clarity, compute attention forward pieces again from cache
    const Qh = this.cache.Qh, Kh=this.cache.Kh, Vh=this.cache.Vh;
    for(let b=0;b<batch;b++){
      for(let h=0;h<this.n_heads;h++){
        const dHead = dHeads[b].map(r=>r[h]); // seq x d_head
        const Qmat = Qh[b].map(r=>r[h]);
        const Kmat = Kh[b].map(r=>r[h]);
        const Vmat = Vh[b].map(r=>r[h]);
        const scale = 1/Math.sqrt(this.d_head);
        // recompute scores and attn
        const scores = Array.from({length:seq},(_,i)=>Array(seq).fill(0));
        for(let i=0;i<seq;i++) for(let j=0;j<seq;j++){
          let s=0;
          for(let d=0;d<this.d_head;d++) s+=Qmat[i][d]*Kmat[j][d];
          scores[i][j]=s*scale;
        }
        for(let i=0;i<seq;i++) for(let j=i+1;j<seq;j++) scores[i][j]=-1e9;
        const attn = softmaxRows(scores);
        // gradients:
        // dV: for each position j, dV[j] += sum_i attn[i][j] * dHead[i]
        for(let j=0;j<seq;j++){
          const accum = Array(this.d_head).fill(0);
          for(let i=0;i<seq;i++){
            const a = attn[i][j];
            for(let d=0;d<this.d_head;d++) accum[d] += a * dHead[i][d];
          }
          dV[b][j].splice(h*this.d_head, this.d_head, ...accum);
        }
        // dAttn[i][j] = sum_d dHead[i,d] * Vmat[j,d]
        const dAttn = Array.from({length:seq},()=>Array(seq).fill(0));
        for(let i=0;i<seq;i++){
          for(let j=0;j<seq;j++){
            let s=0;
            for(let d=0;d<this.d_head;d++) s += dHead[i][d] * Vmat[j][d];
            dAttn[i][j]=s;
          }
        }
        // convert through softmax: for row i, dScores = (J - a a^T) * dAttn_row where J is diag(a)
        const dScores = Array.from({length:seq},()=>Array(seq).fill(0));
        for(let i=0;i<seq;i++){
          const a = attn[i];
          const rowGrad = dAttn[i];
          const ssum = a.reduce((acc,ai,idx)=>acc + ai * rowGrad[idx], 0);
          for(let j=0;j<seq;j++) dScores[i][j] = a[j] * (rowGrad[j] - ssum);
        }
        // scale back
        for(let i=0;i<seq;i++) for(let j=0;j<seq;j++) dScores[i][j] *= scale;
        // dQ[i] += sum_j dScores[i][j] * K[j]
        for(let i=0;i<seq;i++){
          const accum = Array(this.d_head).fill(0);
          for(let j=0;j<seq;j++){
            const s = dScores[i][j];
            for(let d=0;d<this.d_head;d++) accum[d] += s * Kmat[j][d];
          }
          dQ[b][i].splice(h*this.d_head, this.d_head, ...accum);
        }
        // dK[j] += sum_i dScores[i][j] * Q[i]
        for(let j=0;j<seq;j++){
          const accum = Array(this.d_head).fill(0);
          for(let i=0;i<seq;i++){
            const s = dScores[i][j];
            for(let d=0;d<this.d_head;d++) accum[d] += s * Qmat[i][d];
          }
          dK[b][j].splice(h*this.d_head, this.d_head, ...accum);
        }
      }
    }
    // flatten dQ,dK,dV to (batch*seq) x d_model
    const dQflat = dQ.flat();
    const dKflat = dK.flat();
    const dVflat = dV.flat();
    // backward through q,k,v linear layers
    const dXq = this.q.backward(dQflat); // returns (batch*seq) x d_model
    const dXk = this.k.backward(dKflat);
    const dXv = this.v.backward(dVflat);
    // sum contributions to dX (inputs to q,k,v were same flattened X)
    // dXsum = dXq + dXk + dXv
    const dXsum = Array.from({length:dXq.length},(_,i)=>dXq[i].map((v,j)=>v + dXk[i][j] + dXv[i][j]));
    // reshape to batch x seq x d_model
    const dX = Array.from({length:batch},(_,b)=>dXsum.slice(b*seq,(b+1)*seq));
    return dX;
  }
  getParams() {
    return Object.assign({}, this.q.getParamsObj(this.name+'.q'), this.k.getParamsObj(this.name+'.k'),
      this.v.getParamsObj(this.name+'.v'), this.o.getParamsObj(this.name+'.o'));
  }
  getGrads() {
    return {
      [this.name+'.q.dW']: this.q.dW, [this.name+'.q.db']: this.q.db,
      [this.name+'.k.dW']: this.k.dW, [this.name+'.k.db']: this.k.db,
      [this.name+'.v.dW']: this.v.dW, [this.name+'.v.db']: this.v.db,
      [this.name+'.o.dW']: this.o.dW, [this.name+'.o.db']: this.o.db
    };
  }
}

// ---------- FeedForward (MLP) ----------
class FeedForward {
  constructor(d_model, d_ff, name='ffn') {
    this.lin1 = new Linear(d_model, d_ff, name+'.lin1');
    this.lin2 = new Linear(d_ff, d_model, name+'.lin2');
    this.name = name;
  }
  forward(X) {
    // X: batch x seq x d_model -> flatten
    const flat = X.flat();
    this.X = X;
    const h1 = this.lin1.forward(flat); // (batch*seq) x d_ff
    this.h1 = h1;
    // apply GELU
    this.g = h1.map(geluRow);
    const h2 = this.lin2.forward(this.g); // (batch*seq) x d_model
    const out = Array.from({length:X.length},(_,b)=>h2.slice(b*X[0].length,(b+1)*X[0].length));
    return out;
  }
  backward(dY) {
    const flat_dY = dY.flat();
    const dH2 = this.lin2.backward(flat_dY); // (batch*seq) x d_ff
    // back through GELU
    const dG = dH2.map((row,i)=>geluBackward(this.h1[i], row));
    const dXflat = this.lin1.backward(dG);
    const batch = dY.length, seq = dY[0].length;
    const dX = Array.from({length:batch},(_,b)=>dXflat.slice(b*seq,(b+1)*seq));
    return dX;
  }
  getParams() {
    return Object.assign({}, this.lin1.getParamsObj(this.name+'.lin1'), this.lin2.getParamsObj(this.name+'.lin2'));
  }
  getGrads() {
    return {
      [this.name+'.lin1.dW']: this.lin1.dW, [this.name+'.lin1.db']: this.lin1.db,
      [this.name+'.lin2.dW']: this.lin2.dW, [this.name+'.lin2.db']: this.lin2.db
    };
  }
}

// ---------- Transformer Block ----------
class TransformerBlock {
  constructor(d_model, n_heads, d_ff, name='block') {
    this.mha = new MultiHeadAttention(d_model, n_heads, name+'.mha');
    this.ln1 = new LayerNorm(d_model);
    this.ffn = new FeedForward(d_model, d_ff, name+'.ffn');
    this.ln2 = new LayerNorm(d_model);
    this.name=name;
  }
  forward(X) { // X: batch x seq x d_model
    // mha
    const mha_out = this.mha.forward(X); // batch x seq x d_model
    // residual + ln1
    const res1 = add(X, mha_out);
    const ln1_out = this.ln1.forward(res1.map(r=>r)); // layernorm accepts batch x dim but we have seq; flatten seq as batch*seq
    // But our LayerNorm expects batch x dim; so reshape:
    const b = X.length, seq = X[0].length;
    const ln1_in = [];
    for(let i=0;i<b;i++) for(let j=0;j<seq;j++) ln1_in.push(res1[i][j]);
    const ln1_out_flat = this.ln1.forward(ln1_in);
    // reshape back
    const ln1_out_reshaped = Array.from({length:b},(_,i)=>ln1_out_flat.slice(i*seq,(i+1)*seq));
    // ffn
    const ffn_out = this.ffn.forward(ln1_out_reshaped);
    const res2 = add(ln1_out_reshaped, ffn_out);
    // ln2
    const ln2_in = [];
    for(let i=0;i<b;i++) for(let j=0;j<seq;j++) ln2_in.push(res2[i][j]);
    const ln2_out_flat = this.ln2.forward(ln2_in);
    const out = Array.from({length:b},(_,i)=>ln2_out_flat.slice(i*seq,(i+1)*seq));
    this.cache = {X, mha_out, res1, ln1_out_reshaped, ffn_out, res2};
    return out;
  }
  backward(dOut) {
    const b=dOut.length, seq=dOut[0].length;
    // ln2 backward
    const ln2_in = this.cache.res2;
    const ln2_in_flat = ln2_in.flat();
    const dOutFlat = dOut.flat();
    const dln2in_flat = this.ln2.backward(dOutFlat); // returns batch*seq x d_model
    const dln2in = Array.from({length:b},(_,i)=>dln2in_flat.slice(i*seq,(i+1)*seq));
    // through residual: dres2 = dln2in; ffn had res2 = ln1_out + ffn_out
    const dffn_out = dln2in.map((row,idx)=>row.map(v=>v*1)); // same shape
    const dln1_out = dln2in.map((row,idx)=>row.map(v=>v*1));
    // ffn backward
    const dln1_from_ffn = this.ffn.backward(dffn_out);
    // sum gradients to ln1_out
    const dln1_total = dln1_out.map((row,i)=>row.map((v,j)=>v + dln1_from_ffn[i][j]));
    // ln1 backward
    const ln1_in_flat = this.cache.res1.flat();
    const dln1_flat = dln1_total.flat();
    const dres1_flat = this.ln1.backward(dln1_flat);
    const dres1 = Array.from({length:b},(_,i)=>dres1_flat.slice(i*seq,(i+1)*seq));
    // res1 = X + mha_out => so dX += dres1, dMHA += dres1
    const dX_from_block1 = dres1.map(row=>row.map(v=>v));
    const dMHA_in = dres1.map(row=>row.map(v=>v));
    // mha backward
    const dX_from_mha = this.mha.backward(dMHA_in);
    // total dX = dX_from_block1 + dX_from_mha
    const dX = dX_from_block1.map((row,i)=>row.map((v,j)=>v + dX_from_mha[i][j]));
    return dX;
  }
  getParams() {
    return Object.assign({}, this.mha.getParams(), this.ln1.getNamedParams(this.name+'.ln1'), this.ffn.getParams(), this.ln2.getNamedParams(this.name+'.ln2'));
  }
  getGrads() {
    return Object.assign({}, this.mha.getGrads(), this.ffn.getGrads(), {
      [this.name+'.ln1.dg']: this.ln1.dg, [this.name+'.ln1.db']: this.ln1.db,
      [this.name+'.ln2.dg']: this.ln2.dg, [this.name+'.ln2.db']: this.ln2.db
    });
  }
}

// ---------- Full Model ----------
class GPT {
  constructor(tokenizer, config) {
    this.tok = tokenizer;
    this.d_model = config.d_model;
    this.vocab_size = tokenizer.vocab_size;
    this.max_seq_len = config.max_seq_len;
    this.n_layers = config.n_layers;
    this.d_ff = config.d_ff;
    this.n_heads = config.n_heads;
    // embeddings
    this.wte = randn([this.vocab_size, this.d_model], 0.02); // vocab x d_model
    this.wpe = this.makePositionalEncoding(this.max_seq_len, this.d_model);
    // model blocks
    this.blocks = [];
    for(let i=0;i<this.n_layers;i++) this.blocks.push(new TransformerBlock(this.d_model, this.n_heads, this.d_ff, 'block'+i));
    // final layernorm
    this.ln_f = new LayerNorm(this.d_model);
    // tied lm head (we'll use matrix multiply by embedding^T)
    // optimizer states for embedding
    this.m_wte = zeros([this.vocab_size, this.d_model]); this.v_wte = zeros([this.vocab_size, this.d_model]);
    // adam counts
    this.t = 0;
  }
  makePositionalEncoding(maxLen, d_model) {
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
  embedInput(ids) { // ids: batch x seq
    const b = ids.length, seq = ids[0].length;
    const out = Array.from({length:b},(_,i)=>Array.from({length:seq},(_,j)=>{
      const id=ids[i][j]; const emb = this.wte[id] || this.wte[this.tok.vocab['<unk>']];
      const pos = this.wpe[j] || Array(this.d_model).fill(0);
      return emb.map((v,k)=>v + pos[k]);
    }));
    return out;
  }
  forward(ids) {
    // ids: batch x seq
    const x = this.embedInput(ids); // batch x seq x d_model
    let h = x;
    for(const blk of this.blocks) {
      h = blk.forward(h);
    }
    // final ln
    const flat = h.flat();
    const ln_out = this.ln_f.forward(flat);
    this.cache = {h, ln_out, ids};
    const out = Array.from({length:ids.length},(_,i)=>ln_out.slice(i*ids[0].length,(i+1)*ids[0].length));
    // logits = out (batch*seq x d_model) * wte^T -> (batch*seq) x vocab
    const flat_out = out.flat();
    const logits = matmul(flat_out.map(r=>r), transpose(this.wte)); // (batch*seq) x vocab
    const logits_batched = Array.from({length:ids.length},(_,i)=>logits.slice(i*ids[0].length,(i+1)*ids[0].length));
    return logits_batched; // batch x seq x vocab
  }
  computeLossAndGrad(logits, labels) {
    // logits: batch x seq x vocab, labels: batch x seq (ids)
    const b=logits.length, seq=logits[0].length, V=this.vocab_size;
    // flatten
    const flatLogits = logits.flat();
    // compute softmax and cross-entropy
    const probs = [];
    const grads = [];
    let loss=0;
    for(let i=0;i<flatLogits.length;i++){
      const row = flatLogits[i];
      const mx = Math.max(...row);
      const ex = row.map(v=>Math.exp(v-mx));
      const s = ex.reduce((a,b)=>a+b,0);
      const p = ex.map(v=>v/s);
      probs.push(p);
    }
    const flatLabels = labels.flat();
    const dFlat = [];
    for(let i=0;i<probs.length;i++){
      const p=probs[i];
      const y = flatLabels[i];
      const l = -Math.log( Math.max(p[y], 1e-12) );
      loss += l;
      const grad = p.map((v,idx)=> (idx===y ? v-1 : v) );
      dFlat.push(grad);
    }
    loss /= (b*seq);
    // gradient w.r.t. logits averaged
    const scale = 1/(b*seq);
    const dLogits = Array.from({length:b},(_,bi)=>Array.from({length:seq},(_,si)=>dFlat[bi*seq+si].map(v=>v*scale)));
    return {loss, dLogits};
  }
  backward(dLogits) {
    // dLogits: batch x seq x vocab
    const b=dLogits.length, seq=dLogits[0].length;
    // dFlat x vocab -> propagate through tied embedding: logits = out_flat * wte^T
    const flat_dLogits = dLogits.flat();
    // gradient wrt out_flat = dLogits * wte
    const dOutFlat = matmul(flat_dLogits, this.wte); // (b*seq) x d_model
    // back through ln_f
    const dLnFlat = this.ln_f.backward(dOutFlat);
    // reshape into batch x seq x d_model
    const dLn = Array.from({length:b},(_,i)=>dLnFlat.slice(i*seq,(i+1)*seq));
    // backward through blocks in reverse
    let grad = dLn;
    for(let i=this.blocks.length-1;i>=0;i--){
      grad = this.blocks[i].backward(grad);
    }
    // grad now w.r.t input embeddings (x = wte[id] + pos)
    // separate gradients for wte and positional
    // dWTE: for each id in batch seq, add grad
    this.dWTE = zeros([this.vocab_size, this.d_model]);
    this.dWPE = zeros([this.max_seq_len, this.d_model]);
    const ids = this.cache.ids;
    for(let i=0;i<b;i++){
      for(let j=0;j<seq;j++){
        const id = ids[i][j];
        const g = grad[i][j];
        for(let k=0;k<this.d_model;k++){
          this.dWTE[id][k] += g[k];
          this.dWPE[j][k] += g[k];
        }
      }
    }
    // average across batch
    const scale = 1/(b);
    for(let i=0;i<this.vocab_size;i++) for(let k=0;k<this.d_model;k++) this.dWTE[i][k] *= scale;
    for(let i=0;i<this.max_seq_len;i++) for(let k=0;k<this.d_model;k++) this.dWPE[i][k] *= scale;
    return;
  }
  getAllParams() {
    const p = { wte:this.wte, wpe:this.wpe };
    for(let i=0;i<this.blocks.length;i++){
      Object.assign(p, this.blocks[i].getParams());
    }
    Object.assign(p, this.ln_f.getNamedParams('ln_f'));
    return p;
  }
  getAllGrads() {
    const g = { dWTE:this.dWTE, dWPE:this.dWPE };
    for(let i=0;i<this.blocks.length;i++){
      Object.assign(g, this.blocks[i].getGrads());
    }
    Object.assign(g, { 'ln_f.dg': this.ln_f.dg, 'ln_f.db': this.ln_f.db });
    return g;
  }
  // ---------- Optimizer: Adam for all params ----------
  stepAdam(lr, betas, eps) {
    this.t += 1;
    const [b1, b2] = betas;
    // embeddings
    for(let i=0;i<this.vocab_size;i++){
      for(let k=0;k<this.d_model;k++){
        const g = this.dWTE[i][k];
        this.m_wte[i][k] = b1 * this.m_wte[i][k] + (1-b1) * g;
        this.v_wte[i][k] = b2 * this.v_wte[i][k] + (1-b2) * g*g;
        const mhat = this.m_wte[i][k] / (1 - Math.pow(b1, this.t));
        const vhat = this.v_wte[i][k] / (1 - Math.pow(b2, this.t));
        this.wte[i][k] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
    }
    // positional encodings - treat as parameters to update
    if (!this.m_wpe) { this.m_wpe = zeros([this.max_seq_len, this.d_model]); this.v_wpe = zeros([this.max_seq_len, this.d_model]); }
    for(let i=0;i<this.max_seq_len;i++){
      for(let k=0;k<this.d_model;k++){
        const g = this.dWPE[i][k];
        this.m_wpe[i][k] = b1*this.m_wpe[i][k] + (1-b1)*g;
        this.v_wpe[i][k] = b2*this.v_wpe[i][k] + (1-b2)*g*g;
        const mhat = this.m_wpe[i][k] / (1 - Math.pow(b1, this.t));
        const vhat = this.v_wpe[i][k] / (1 - Math.pow(b2, this.t));
        this.wpe[i][k] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
    }
    // now update all submodules (Linear layers, LayerNorms) using their stored grads & adam states
    // helper update for a linear
    function updateLinear(L, namePrefix) {
      const W=L.W, b=L.b;
      const dW=L.dW, db=L.db;
      for(let i=0;i<W.length;i++) for(let j=0;j<W[0].length;j++){
        L.mW[i][j] = b1*L.mW[i][j] + (1-b1)*dW[i][j];
        L.vW[i][j] = b2*L.vW[i][j] + (1-b2)*dW[i][j]*dW[i][j];
        const mhat = L.mW[i][j] / (1 - Math.pow(b1, this.t));
        const vhat = L.vW[i][j] / (1 - Math.pow(b2, this.t));
        W[i][j] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
      for(let j=0;j<b.length;j++){
        L.mb[j] = b1*L.mb[j] + (1-b1)*db[j];
        L.vb[j] = b2*L.vb[j] + (1-b2)*db[j]*db[j];
        const mhat = L.mb[j] / (1 - Math.pow(b1, this.t));
        const vhat = L.vb[j] / (1 - Math.pow(b2, this.t));
        b[j] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
    }
    // update LayerNorm
    function updateLayerNorm(LN) {
      const g=LN.g, b=LN.b, dg=LN.dg, db=LN.db;
      if (!LN.mg) { LN.mg = zeros([g.length]); LN.vg = zeros([g.length]); LN.mb=zeros([b.length]); LN.vb=zeros([b.length]); }
      for(let i=0;i<g.length;i++){
        LN.mg[i] = b1*LN.mg[i] + (1-b1)*dg[i];
        LN.vg[i] = b2*LN.vg[i] + (1-b2)*dg[i]*dg[i];
        const mg_hat = LN.mg[i]/(1-Math.pow(b1,this.t));
        const vg_hat = LN.vg[i]/(1-Math.pow(b2,this.t));
        g[i] -= lr * mg_hat / (Math.sqrt(vg_hat)+eps);
        LN.mb[i] = b1*LN.mb[i] + (1-b1)*db[i];
        LN.vb[i] = b2*LN.vb[i] + (1-b2)*db[i]*db[i];
        const mb_hat = LN.mb[i]/(1-Math.pow(b1,this.t));
        const vb_hat = LN.vb[i]/(1-Math.pow(b2,this.t));
        b[i] -= lr * mb_hat / (Math.sqrt(vb_hat)+eps);
      }
    }
    // walk through blocks and update contained linears & layernorms
    for(const blk of this.blocks){
      // mha
      updateLinear.call(this, blk.mha.q, blk.mha.name+'.q');
      updateLinear.call(this, blk.mha.k, blk.mha.name+'.k');
      updateLinear.call(this, blk.mha.v, blk.mha.name+'.v');
      updateLinear.call(this, blk.mha.o, blk.mha.name+'.o');
      // ln1 ln2
      updateLayerNorm(blk.ln1);
      updateLayerNorm(blk.ln2);
      // ffn
      updateLinear.call(this, blk.ffn.lin1, blk.ffn.name+'.lin1');
      updateLinear.call(this, blk.ffn.lin2, blk.ffn.name+'.lin2');
    }
    // ln_f
    updateLayerNorm(this.ln_f);
  }
  save(path) {
    const data = {config:CONFIG, vocab:this.tok.id2tok, wte:this.wte, wpe:this.wpe};
    // Also save block params (linears and layernorm parameters)
    for(let i=0;i<this.blocks.length;i++){
      const blk = this.blocks[i];
      const bname = 'block'+i;
      // mha
      ['q','k','v','o'].forEach(k=>{
        const L = blk.mha[k];
        data[`${bname}.${k}.W`] = L.W;
        data[`${bname}.${k}.b`] = L.b;
      });
      // ffn
      data[`${bname}.ffn.lin1.W`] = blk.ffn.lin1.W;
      data[`${bname}.ffn.lin1.b`] = blk.ffn.lin1.b;
      data[`${bname}.ffn.lin2.W`] = blk.ffn.lin2.W;
      data[`${bname}.ffn.lin2.b`] = blk.ffn.lin2.b;
      // ln params
      data[`${bname}.ln1.g`] = blk.ln1.g; data[`${bname}.ln1.b`] = blk.ln1.b;
      data[`${bname}.ln2.g`] = blk.ln2.g; data[`${bname}.ln2.b`] = blk.ln2.b;
    }
    data['ln_f.g']=this.ln_f.g; data['ln_f.b']=this.ln_f.b;
    fs.writeFileSync(path, JSON.stringify(data));
    console.log('Saved weights to', path);
  }
  load(path) {
    const raw = fs.readFileSync(path,'utf8'); const data = JSON.parse(raw);
    if (data.wte) this.wte = data.wte;
    if (data.wpe) this.wpe = data.wpe;
    // load blocks
    for(let i=0;i<this.blocks.length;i++){
      const bname='block'+i; const blk=this.blocks[i];
      ['q','k','v','o'].forEach(k=>{
        const W = data[`${bname}.${k}.W`]; const b = data[`${bname}.${k}.b`];
        if (W) blk.mha[k].W = W;
        if (b) blk.mha[k].b = b;
      });
      const l1W=data[`${bname}.ffn.lin1.W`], l1b=data[`${bname}.ffn.lin1.b`];
      const l2W=data[`${bname}.ffn.lin2.W`], l2b=data[`${bname}.ffn.lin2.b`];
      if (l1W) blk.ffn.lin1.W = l1W; if (l1b) blk.ffn.lin1.b = l1b;
      if (l2W) blk.ffn.lin2.W = l2W; if (l2b) blk.ffn.lin2.b = l2b;
      const ln1g=data[`${bname}.ln1.g`], ln1b=data[`${bname}.ln1.b`];
      const ln2g=data[`${bname}.ln2.g`], ln2b=data[`${bname}.ln2.b`];
      if (ln1g) blk.ln1.g = ln1g; if (ln1b) blk.ln1.b = ln1b;
      if (ln2g) blk.ln2.g = ln2g; if (ln2b) blk.ln2.b = ln2b;
    }
    if (data['ln_f.g']) this.ln_f.g = data['ln_f.g'];
    if (data['ln_f.b']) this.ln_f.b = data['ln_f.b'];
    console.log('Loaded weights from', path);
  }
  // ---------- Sampling: top-k + temperature ----------
  sample(prefix_ids, max_new_tokens=20, temperature=1.0, top_k=50) {
    let ids = prefix_ids.slice();
    for(let t=0;t<max_new_tokens;t++){
      // prepare input of length <= max_seq_len
      const context = ids.slice(-this.max_seq_len);
      while (context.length < this.max_seq_len) context.unshift(this.tok.vocab['<pad>']);
      const logits = this.forward([context]); // shape [1][seq][vocab]
      const lastLogits = logits[0][context.length-1];
      // apply temperature
      const scaled = lastLogits.map(v=>v/temperature);
      // top-k
      const idxs = scaled.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,top_k);
      const exps = idxs.map(x=>Math.exp(x[0]-Math.max(...idxs.map(y=>y[0]))));
      const s = exps.reduce((a,b)=>a+b,0);
      const probs = exps.map(v=>v/s);
      // sample from probs
      let r=Math.random(), cum=0, pick=idxs[0][1];
      for(let i=0;i<probs.length;i++){
        cum+=probs[i];
        if (r < cum) { pick = idxs[i][1]; break; }
      }
      ids.push(pick);
      if (pick === this.tok.vocab['<eos>']) break;
      if (ids.length > this.max_seq_len*2) break;
    }
    return ids;
  }
}

// ---------- Example: prepare small dataset and train ----------
function exampleRun() {
  // small toy dataset
  const texts = [
    "hello world this is a test",
    "hello there how are you",
    "this is another example text",
    "the model will learn to predict words"
  ];
  // build tokenizer
  const tok = new Tokenizer();
  for(const t of texts) tok.feedText(t);
  tok.buildVocab(1, 20000);
  console.log('Vocab size:', tok.vocab_size);
  // create model
  const cfg = {
    d_model: CONFIG.d_model, n_heads: CONFIG.n_heads, d_ff: CONFIG.d_ff,
    n_layers: CONFIG.n_layers, max_seq_len: CONFIG.max_seq_len
  };
  const model = new GPT(tok, cfg);
  // prepare training pairs (simple next-token prediction)
  const examples = texts.map(t=>tok.encode(t, CONFIG.max_seq_len));
  // training loop (very small)
  const epochs = 3;
  for(let ep=0; ep<epochs; ep++){
    let epochLoss = 0;
    for(const ex of examples){
      const ids = [ex];
      const logits = model.forward(ids);
      const {loss, dLogits} = model.computeLossAndGrad(logits, ids);
      epochLoss += loss;
      model.backward(dLogits);
      model.stepAdam(CONFIG.lr, CONFIG.betas, CONFIG.eps);
    }
    console.log(`Epoch ${ep+1}/${epochs} loss=${epochLoss/examples.length}`);
    model.save(CONFIG.save_path);
  }
  // sample
  const prefix = tok.encode("hello", CONFIG.max_seq_len).slice(0,CONFIG.max_seq_len);
  const gen = model.sample(prefix, 10, 1.0, 10);
  console.log('Generated ids:', gen);
  console.log('Generated text:', tok.decode(gen));
}

// Run example if main
if (require.main === module) {
  exampleRun();
}
