/* ─── VoiceGuard Core ─── */
const params = new URLSearchParams(window.location.search);
const API = params.get('api') || 'http://localhost:8000';
let currentFile = null, apiLive = false, analysisHistory = [], currentMode = 'upload', sessionStats = {total:0,fake:0,real:0};

/* ─── Health check ─── */
const hc = new AbortController();
const ht = setTimeout(() => hc.abort(), 4000);
fetch(`${API}/health`, {signal:hc.signal})
  .then(r => {clearTimeout(ht); if(r.ok){apiLive=true;dot('live');txt('Inference server online — real-time detection active')}else throw 0})
  .catch(() => {clearTimeout(ht);dot('offline');txt('Server offline — demo simulation mode');show('apiWarn')});

/* ─── DOM helpers ─── */
const el = id => document.getElementById(id);
const show = id => el(id)?.classList.add('show');
const hide = id => el(id)?.classList.remove('show');
function dot(c){el('statusDot').className='status-dot '+c}
function txt(t){el('statusText').textContent=t}
function showError(msg){const e=el('errorMsg');e.textContent='⚠ '+msg;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),6000)}

/* ─── Mode switching ─── */
function switchMode(mode){
  currentMode=mode;
  document.querySelectorAll('.mode-tab').forEach(t=>t.classList.toggle('active',t.dataset.mode===mode));
  el('uploadSection').style.display=mode==='upload'?'block':'none';
  el('micSection').style.display=mode==='mic'?'block':'none';
  if(mode==='mic')initMic();
}

/* ─── Drag & drop ─── */
function initDropzone(){
  const dz=el('dropzone');
  dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
  dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
  dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');if(e.dataTransfer.files[0])loadFile(e.dataTransfer.files[0])});
  el('fileInput').addEventListener('change',e=>{if(e.target.files[0])loadFile(e.target.files[0])});
}

function loadFile(file){
  if(file.size>50*1024*1024){showError(`File too large (${(file.size/1024/1024).toFixed(1)} MB). Max: 50 MB`);return}
  currentFile=file;
  const url=URL.createObjectURL(file);
  showWavePanel(url,file.name);
}

/* ─── Samples ─── */
const SAMPLES={real_1:{name:'human_voice_01.wav',label:'REAL'},real_2:{name:'human_voice_02.wav',label:'REAL'},fake_1:{name:'elevenlabs_clone.wav',label:'FAKE'},fake_2:{name:'tts_synthetic.wav',label:'FAKE'}};
function loadSample(id){
  currentFile={name:SAMPLES[id].name,_demo:SAMPLES[id].label,size:80000};
  showWavePanel(null,SAMPLES[id].name);
  drawDemoWaveform(id.startsWith('real'));
}

/* ─── Waveform ─── */
function showWavePanel(url,name){
  el('waveFilename').textContent='📂 '+name;
  const p=el('player');
  if(url){p.src=url;p.style.display='';drawRealWaveform(url)}else{p.style.display='none'}
  el('wavePanel').classList.add('show');
  el('analyzeBtn').classList.add('show');
  hide('result');hide('processing');
  el('result').className='result';
}
async function drawRealWaveform(url){
  try{const ctx=new(window.AudioContext||window.webkitAudioContext)();const buf=await(await fetch(url)).arrayBuffer();const d=await ctx.decodeAudioData(buf);renderWaveform(d.getChannelData(0))}catch{drawDemoWaveform(true)}
}
function drawDemoWaveform(isReal){
  const d=new Float32Array(1000);
  for(let i=0;i<d.length;i++)d[i]=isReal?(Math.sin(i*.13)*.6+Math.sin(i*.31)*.3)*(.5+.5*Math.random()):(Math.sin(i*.11)*.8)*(.85+.15*Math.random());
  renderWaveform(d);
}
function renderWaveform(data){
  const c=el('waveform'),dpr=window.devicePixelRatio||1,W=(c.offsetWidth||800)*dpr,H=140;
  c.width=W;c.height=H;const ctx=c.getContext('2d');ctx.clearRect(0,0,W,H);
  const step=Math.ceil(data.length/W),g=ctx.createLinearGradient(0,0,W,0);
  g.addColorStop(0,'rgba(0,210,255,0.3)');g.addColorStop(.5,'rgba(0,210,255,0.8)');g.addColorStop(1,'rgba(0,210,255,0.3)');
  ctx.strokeStyle=g;ctx.lineWidth=1.2;ctx.beginPath();
  for(let i=0;i<W;i++){let mn=1,mx=-1;for(let j=0;j<step;j++){const v=data[i*step+j]||0;if(v<mn)mn=v;if(v>mx)mx=v}ctx.moveTo(i,(1+mn)*H/2);ctx.lineTo(i,Math.max(1,(1+mx)*H/2))}
  ctx.stroke();ctx.strokeStyle='rgba(0,210,255,0.08)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
}

/* ─── Processing animation ─── */
const STEPS=['step0','step1','step2','step3','step4','step5'];
let stepTimer=null;
function startStepAnim(){let i=0;STEPS.forEach(s=>el(s).className='proc-step');stepTimer=setInterval(()=>{if(i>0)el(STEPS[i-1]).className='proc-step done';if(i<STEPS.length){el(STEPS[i]).className='proc-step active';i++}else clearInterval(stepTimer)},420)}
function endStepAnim(){clearInterval(stepTimer);STEPS.forEach(s=>el(s).className='proc-step done')}

/* ─── Analyze ─── */
async function analyze(){
  if(!currentFile)return;
  const btn=el('analyzeBtn');btn.disabled=true;hide('errorMsg');show('processing');hide('result');startStepAnim();
  let data;
  if(apiLive&&currentFile instanceof File){
    const form=new FormData();form.append('file',currentFile);
    try{const r=await fetch(`${API}/predict`,{method:'POST',body:form});data=await r.json();if(!r.ok)throw new Error(data.detail||JSON.stringify(data))}
    catch(err){endStepAnim();hide('processing');btn.disabled=false;showError('Server error: '+err.message);data=simulateResult(currentFile.name)}
  }else{await new Promise(r=>setTimeout(r,2800));data=simulateResult(currentFile._demo||currentFile.name)}
  endStepAnim();await new Promise(r=>setTimeout(r,300));hide('processing');
  if(data){showResult(data);addHistory(data);updateDashboard(data)}
  btn.disabled=false;
}

/* ─── Simulate ─── */
function simulateResult(hint){
  const h=typeof hint==='string'?hint.toLowerCase():'';
  const isFake=hint==='FAKE'||h.includes('clone')||h.includes('fake')||h.includes('tts')||h.includes('synthetic')||h.includes('eleven');
  const fs=isFake?.74+Math.random()*.2:.04+Math.random()*.16;const rs=1-fs;const label=fs>.5?'FAKE':'REAL';
  return{label,fake_score:+fs.toFixed(4),real_score:+rs.toFixed(4),confidence:+Math.max(fs,rs).toFixed(4),features:270,
    processing_time_ms:120+Math.floor(Math.random()*80),filename:(currentFile&&currentFile.name)||'demo.wav',
    verdict_reason:isFake?`Flagged as synthetic. Key signal: Spectral flatness. Confidence ${Math.round(fs*100)}% — patterns match AI-generated speech.`:`Likely authentic human speech. Key signal: Pitch variation. Confidence ${Math.round(rs*100)}% — natural prosody consistent with real voice.`,
    top_features:isFake?[
      {name:'Spectral flatness',key:'spectral_flatness',value:0.312,weight:18.4},
      {name:'Pitch variation (f0_std)',key:'f0_std',value:2.1,weight:14.2},
      {name:'Voiced ratio',key:'voiced_ratio',value:0.68,weight:11.7},
      {name:'Spectral centroid',key:'spectral_centroid',value:1842,weight:9.8},
      {name:'Zero-crossing rate',key:'zcr',value:0.091,weight:8.3}
    ]:[
      {name:'Pitch variation (f0_std)',key:'f0_std',value:18.4,weight:16.1},
      {name:'Voiced ratio',key:'voiced_ratio',value:0.82,weight:13.5},
      {name:'Spectral flatness',key:'spectral_flatness',value:0.041,weight:10.2},
      {name:'Spectral centroid',key:'spectral_centroid',value:2210,weight:9.1},
      {name:'Zero-crossing rate',key:'zcr',value:0.074,weight:7.8}
    ],
    segments: generateDemoSegments(isFake)
  };
}

function generateDemoSegments(isFake){
  const segs=[];const dur=3.0;const step=0.25;
  for(let t=0;t<dur;t+=step){
    let fs;
    if(isFake){fs=0.65+Math.random()*0.3+0.05*Math.sin(t*4)}
    else{fs=0.05+Math.random()*0.15+0.03*Math.sin(t*3)}
    segs.push({start_sec:+t.toFixed(2),end_sec:+(t+step).toFixed(2),fake_score:+fs.toFixed(4),real_score:+(1-fs).toFixed(4)});
  }
  return segs;
}

/* ─── Show result ─── */
function showResult(data){
  const isFake=data.label==='FAKE',color=isFake?'var(--red)':'var(--green)';
  el('result').className='result show';
  el('verdictBanner').className=`verdict-banner ${isFake?'fake':'real'}`;
  el('verdictWord').textContent=data.label;
  el('verdictWord').className=`verdict-word ${isFake?'fake':'real'}`;
  el('verdictSub').textContent=isFake?'Synthetic / AI-cloned speech detected':'Authentic human voice — no spoofing detected';
  el('verdictReason').textContent=data.verdict_reason||'';
  el('verdictReason').className=`verdict-reason ${isFake?'fake':''}`;

  const cp=Math.round(data.confidence*100);
  el('confPct').textContent=cp+'%';el('confPct').style.color=color;
  const arc=el('confArc');arc.style.stroke=color;
  setTimeout(()=>arc.style.strokeDashoffset=245*(1-data.confidence),50);

  const rP=Math.round(data.real_score*100),fP=Math.round(data.fake_score*100);
  el('realPct').textContent=rP+'%';el('fakePct').textContent=fP+'%';
  setTimeout(()=>{el('realBar').style.width=rP+'%';el('fakeBar').style.width=fP+'%'},100);

  el('statConf').textContent=cp+'%';el('statFeats').textContent=data.features;
  el('statMs').textContent=data.processing_time_ms;
  const fn=data.filename||'';const di=fn.lastIndexOf('.');
  el('statFmt').textContent=di>=0?fn.slice(di+1).toUpperCase():'—';

  const feats=data.top_features||[];
  if(feats.length){
    const mW=Math.max(...feats.map(f=>f.weight));
    el('featRows').innerHTML=feats.map(f=>`<div class="feat-row"><div class="feat-meta"><div class="feat-name">${f.name}</div><div class="feat-badge">${f.weight}% · val: ${f.value}</div></div><div class="feat-track"><div class="feat-fill" id="ff_${f.key}" style="width:0%"></div></div></div>`).join('');
    setTimeout(()=>feats.forEach(f=>{const e=document.getElementById('ff_'+f.key);if(e)e.style.width=Math.round(f.weight/mW*100)+'%'}),150);
    drawRadar(feats,isFake);
  }
  drawSpectral(isFake);
  drawTimeline(data.segments||generateDemoSegments(isFake),isFake);
  drawDNA(data,isFake);
}

/* ─── Radar ─── */
function drawRadar(feats,isFake){
  const c=el('radarChart'),W=240,H=180,cx=W/2,cy=H/2,r=70;c.width=W;c.height=H;
  const ctx=c.getContext('2d'),n=feats.length,angles=feats.map((_,i)=>(i/n)*Math.PI*2-Math.PI/2),max=Math.max(...feats.map(f=>f.weight));
  [.25,.5,.75,1].forEach(t=>{ctx.beginPath();angles.forEach((a,i)=>{const x=cx+Math.cos(a)*r*t,y=cy+Math.sin(a)*r*t;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});ctx.closePath();ctx.strokeStyle='rgba(0,210,255,0.08)';ctx.lineWidth=1;ctx.stroke()});
  angles.forEach(a=>{ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.cos(a)*r,cy+Math.sin(a)*r);ctx.strokeStyle='rgba(0,210,255,0.08)';ctx.lineWidth=1;ctx.stroke()});
  const col=isFake?'255,51,102':'0,255,170';
  ctx.beginPath();feats.forEach((f,i)=>{const t=f.weight/max,x=cx+Math.cos(angles[i])*r*t,y=cy+Math.sin(angles[i])*r*t;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
  ctx.closePath();ctx.fillStyle=`rgba(${col},0.12)`;ctx.fill();ctx.strokeStyle=`rgba(${col},0.8)`;ctx.lineWidth=1.5;ctx.stroke();
  feats.forEach((f,i)=>{const t=f.weight/max,x=cx+Math.cos(angles[i])*r*t,y=cy+Math.sin(angles[i])*r*t;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle=`rgb(${col})`;ctx.fill()});
  ctx.font='9px JetBrains Mono,monospace';ctx.fillStyle='rgba(180,210,240,0.6)';ctx.textAlign='center';
  feats.forEach((f,i)=>{const x=cx+Math.cos(angles[i])*(r+18),y=cy+Math.sin(angles[i])*(r+18)+3;ctx.fillText(f.name.split('(')[0].trim().split(' ').slice(-1)[0],x,y)});
}

/* ─── Spectral ─── */
function drawSpectral(isFake){
  const c=el('spectralViz'),dpr=window.devicePixelRatio||1,W=(c.offsetWidth||800)*dpr,H=100*dpr;
  c.width=W;c.height=H;const ctx=c.getContext('2d');ctx.clearRect(0,0,W,H);
  const bars=Math.floor(W/(3*dpr));
  for(let i=0;i<bars;i++){const x=i*3*dpr;let e=isFake?.5+.45*Math.sin(i*.18)+.05*(Math.random()-.5):.3+.4*Math.abs(Math.sin(i*.13+Math.sin(i*.07)*2))+.2*Math.random();
    e=Math.max(.05,Math.min(1,e));const h=e*(H-4*dpr);
    ctx.fillStyle=isFake?`rgba(255,51,102,${.3+e*.5})`:`rgba(0,210,255,${.25+e*.5})`;ctx.fillRect(x,H-h,2*dpr,h)}
}

/* ─── History ─── */
function addHistory(data){const now=new Date();const time=now.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'});analysisHistory.unshift({...data,time});renderHistory();show('historyCard')}
function renderHistory(){
  el('historyList').innerHTML=analysisHistory.slice(0,8).map(h=>{const iF=h.label==='FAKE',c=Math.round(h.confidence*100),fn=(h.filename||'unknown').substring(0,28);
    return`<div class="history-item"><span class="history-tag ${iF?'fake':'real'}">${h.label}</span><span class="history-fname">${fn}</span><span class="history-conf">${c}%</span><span class="history-time">${h.time}</span></div>`}).join('');
}
function clearHistory(){analysisHistory=[];el('historyList').innerHTML='';hide('historyCard')}

/* ─── Reset ─── */
function resetAll(){
  currentFile=null;el('wavePanel').classList.remove('show');el('analyzeBtn').classList.remove('show');el('analyzeBtn').disabled=false;
  hide('result');hide('processing');hide('errorMsg');el('result').className='result';el('fileInput').value='';el('player').src='';
  el('confArc').style.strokeDashoffset='245';el('realBar').style.width='0%';el('fakeBar').style.width='0%';
}

/* ─── Init ─── */
document.addEventListener('DOMContentLoaded',()=>{initDropzone();initParticles();initKeyboard();initReveal()});
