/* ─── VoiceGuard Features: Mic, DNA, Timeline, Dashboard, Export, Particles, Keyboard ─── */

/* ═══ 1. LIVE MICROPHONE ═══ */
let mediaRecorder, micStream, micChunks=[], micAnalyser, micAnimFrame, micStartTime, micTimerInterval;
function initMic(){
  const btn=el('micBtn');if(!btn)return;
  btn.onclick=async()=>{
    if(mediaRecorder&&mediaRecorder.state==='recording'){stopMicRecording();return}
    try{
      micStream=await navigator.mediaDevices.getUserMedia({audio:true});
      mediaRecorder=new MediaRecorder(micStream,{mimeType:'audio/webm'});
      micChunks=[];
      mediaRecorder.ondataavailable=e=>{if(e.data.size>0)micChunks.push(e.data)};
      mediaRecorder.onstop=()=>{
        const blob=new Blob(micChunks,{type:'audio/webm'});
        currentFile=new File([blob],'mic_recording.webm',{type:'audio/webm'});
        const url=URL.createObjectURL(blob);
        switchMode('upload');showWavePanel(url,'🎙 mic_recording.webm');
        micStream.getTracks().forEach(t=>t.stop());
      };
      mediaRecorder.start();btn.classList.add('recording');
      micStartTime=Date.now();
      el('micTimer').textContent='00:00';
      micTimerInterval=setInterval(()=>{const s=Math.floor((Date.now()-micStartTime)/1000);el('micTimer').textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`},1000);
      el('micHint').textContent='Click to stop recording';
      startMicViz();
    }catch(e){showError('Microphone access denied: '+e.message)}
  };
}
function stopMicRecording(){
  if(mediaRecorder&&mediaRecorder.state==='recording'){mediaRecorder.stop();el('micBtn').classList.remove('recording');clearInterval(micTimerInterval);cancelAnimationFrame(micAnimFrame);el('micHint').textContent='Click to start recording'}
}
function startMicViz(){
  const actx=new(window.AudioContext||window.webkitAudioContext)();
  const src=actx.createMediaStreamSource(micStream);
  micAnalyser=actx.createAnalyser();micAnalyser.fftSize=256;
  src.connect(micAnalyser);
  const c=el('micWaveform'),ctx=c.getContext('2d');
  const dpr=window.devicePixelRatio||1;c.width=(c.offsetWidth||400)*dpr;c.height=120;
  const bufLen=micAnalyser.frequencyBinCount,dataArr=new Uint8Array(bufLen);
  function draw(){
    micAnimFrame=requestAnimationFrame(draw);
    micAnalyser.getByteTimeDomainData(dataArr);
    ctx.fillStyle='rgba(16,28,40,0.3)';ctx.fillRect(0,0,c.width,c.height);
    ctx.lineWidth=2;ctx.strokeStyle='#ff3366';ctx.beginPath();
    const sliceW=c.width/bufLen;let x=0;
    for(let i=0;i<bufLen;i++){const v=dataArr[i]/128.0;const y=v*c.height/2;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);x+=sliceW}
    ctx.lineTo(c.width,c.height/2);ctx.stroke();
  }
  draw();
}

/* ═══ 2. AUDIO DNA FINGERPRINT ═══ */
function drawDNA(data,isFake){
  const c=el('dnaCanvas');if(!c)return;
  const W=220,H=220;c.width=W;c.height=H;
  const ctx=c.getContext('2d'),cx=W/2,cy=H/2;
  const feats=data.top_features||[];
  const conf=data.confidence||0.5;
  const col=isFake?[255,51,102]:[0,255,170];

  // Outer ring
  ctx.beginPath();ctx.arc(cx,cy,95,0,Math.PI*2);
  ctx.strokeStyle=`rgba(${col},0.15)`;ctx.lineWidth=1;ctx.stroke();

  // DNA strands - procedural from feature data
  const strands=36;
  for(let s=0;s<3;s++){
    ctx.beginPath();
    for(let i=0;i<=strands;i++){
      const angle=(i/strands)*Math.PI*2-Math.PI/2;
      const feat=feats[i%feats.length]||{weight:10};
      const baseR=30+s*20;
      const wobble=isFake?3:8+Math.sin(i*1.7)*4;
      const r=baseR+feat.weight*wobble/10+(isFake?Math.sin(i*0.8)*2:Math.sin(i*2.1+s)*6+Math.cos(i*0.7)*4);
      const x=cx+Math.cos(angle)*r,y=cy+Math.sin(angle)*r;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.fillStyle=`rgba(${col},${0.03+s*0.02})`;ctx.fill();
    ctx.strokeStyle=`rgba(${col},${0.2+s*0.15})`;ctx.lineWidth=1.5;ctx.stroke();
  }

  // Data points
  feats.forEach((f,i)=>{
    const angle=(i/feats.length)*Math.PI*2-Math.PI/2;
    const r=40+f.weight*3;
    const x=cx+Math.cos(angle)*r,y=cy+Math.sin(angle)*r;
    ctx.beginPath();ctx.arc(x,y,3+f.weight/8,0,Math.PI*2);
    ctx.fillStyle=`rgba(${col},0.6)`;ctx.fill();
    // Connecting lines to center
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);
    ctx.strokeStyle=`rgba(${col},0.08)`;ctx.lineWidth=0.5;ctx.stroke();
  });

  // Center score
  ctx.beginPath();ctx.arc(cx,cy,22,0,Math.PI*2);
  ctx.fillStyle=`rgba(${col},0.1)`;ctx.fill();
  ctx.strokeStyle=`rgba(${col},0.4)`;ctx.lineWidth=1.5;ctx.stroke();
  ctx.font='bold 14px JetBrains Mono';ctx.fillStyle=`rgb(${col})`;ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillText(Math.round(conf*100)+'%',cx,cy);

  el('dnaLabel').textContent=isFake?'SYNTHETIC SIGNATURE — high symmetry detected':'ORGANIC SIGNATURE — natural asymmetry detected';
}

/* ═══ 3. TEMPORAL CONFIDENCE TIMELINE ═══ */
function drawTimeline(segments,isFake){
  const c=el('timelineChart');if(!c||!segments||!segments.length)return;
  const dpr=window.devicePixelRatio||1;
  const W=(c.offsetWidth||700)*dpr,H=120*dpr;
  c.width=W;c.height=H;
  const ctx=c.getContext('2d');ctx.clearRect(0,0,W,H);
  const pad={l:40*dpr,r:10*dpr,t:15*dpr,b:25*dpr};
  const gW=W-pad.l-pad.r,gH=H-pad.t-pad.b;
  const maxT=segments[segments.length-1].end_sec;

  // Threshold line at 0.5
  const threshY=pad.t+gH*0.5;
  ctx.strokeStyle='rgba(255,255,255,0.1)';ctx.lineWidth=1;ctx.setLineDash([4*dpr,4*dpr]);
  ctx.beginPath();ctx.moveTo(pad.l,threshY);ctx.lineTo(W-pad.r,threshY);ctx.stroke();ctx.setLineDash([]);
  ctx.font=`${8*dpr}px JetBrains Mono`;ctx.fillStyle='rgba(255,255,255,0.2)';ctx.textAlign='right';
  ctx.fillText('FAKE',pad.l-4*dpr,pad.t+10*dpr);ctx.fillText('REAL',pad.l-4*dpr,H-pad.b-2*dpr);

  // Area fill
  ctx.beginPath();
  segments.forEach((s,i)=>{
    const x=pad.l+(s.start_sec/maxT)*gW;
    const y=pad.t+gH*(1-s.fake_score);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  const lastSeg=segments[segments.length-1];
  ctx.lineTo(pad.l+(lastSeg.end_sec/maxT)*gW,pad.t+gH*(1-lastSeg.fake_score));
  ctx.lineTo(pad.l+gW,H-pad.b);ctx.lineTo(pad.l,H-pad.b);ctx.closePath();
  const grd=ctx.createLinearGradient(0,pad.t,0,H-pad.b);
  grd.addColorStop(0,'rgba(255,51,102,0.15)');grd.addColorStop(0.5,'rgba(255,255,255,0.02)');grd.addColorStop(1,'rgba(0,255,170,0.1)');
  ctx.fillStyle=grd;ctx.fill();

  // Line
  ctx.beginPath();
  segments.forEach((s,i)=>{
    const x=pad.l+(s.start_sec/maxT)*gW;
    const y=pad.t+gH*(1-s.fake_score);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.lineTo(pad.l+(lastSeg.end_sec/maxT)*gW,pad.t+gH*(1-lastSeg.fake_score));
  const lineGrd=ctx.createLinearGradient(0,pad.t,0,H-pad.b);
  lineGrd.addColorStop(0,'#ff3366');lineGrd.addColorStop(0.5,'#ffb700');lineGrd.addColorStop(1,'#00ffaa');
  ctx.strokeStyle=lineGrd;ctx.lineWidth=2*dpr;ctx.stroke();

  // Dots
  segments.forEach(s=>{
    const x=pad.l+(s.start_sec/maxT)*gW;
    const y=pad.t+gH*(1-s.fake_score);
    ctx.beginPath();ctx.arc(x,y,3*dpr,0,Math.PI*2);
    ctx.fillStyle=s.fake_score>0.5?'#ff3366':'#00ffaa';ctx.fill();
  });

  // Time labels
  ctx.font=`${8*dpr}px JetBrains Mono`;ctx.fillStyle='rgba(255,255,255,0.3)';ctx.textAlign='center';
  for(let t=0;t<=maxT;t+=0.5){
    const x=pad.l+(t/maxT)*gW;
    ctx.fillText(t.toFixed(1)+'s',x,H-4*dpr);
  }
}

/* ═══ 4. THREAT INTELLIGENCE DASHBOARD ═══ */
function updateDashboard(data){
  sessionStats.total++;
  if(data.label==='FAKE')sessionStats.fake++;else sessionStats.real++;
  el('dashTotal').textContent=sessionStats.total;
  el('dashFakeRate').textContent=sessionStats.total?Math.round(sessionStats.fake/sessionStats.total*100)+'%':'0%';
  el('dashFakeRate').style.color=sessionStats.fake/sessionStats.total>0.5?'var(--red)':'var(--green)';
  drawDashPie();drawThreatGauge();
  el('dashboardCard').style.display='block';
}
function drawDashPie(){
  const c=el('dashPie');if(!c)return;c.width=100;c.height=100;
  const ctx=c.getContext('2d'),cx=50,cy=50,r=38;
  const total=sessionStats.total||1;
  const fakeAngle=(sessionStats.fake/total)*Math.PI*2;
  // Real slice
  ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,0,Math.PI*2-fakeAngle);ctx.closePath();
  ctx.fillStyle='rgba(0,255,170,0.3)';ctx.fill();ctx.strokeStyle='#00ffaa';ctx.lineWidth=1;ctx.stroke();
  // Fake slice
  ctx.beginPath();ctx.moveTo(cx,cy);ctx.arc(cx,cy,r,Math.PI*2-fakeAngle,Math.PI*2);ctx.closePath();
  ctx.fillStyle='rgba(255,51,102,0.3)';ctx.fill();ctx.strokeStyle='#ff3366';ctx.lineWidth=1;ctx.stroke();
  // Center
  ctx.beginPath();ctx.arc(cx,cy,18,0,Math.PI*2);ctx.fillStyle='#080d14';ctx.fill();
  ctx.font='bold 11px JetBrains Mono';ctx.fillStyle='#ddeeff';ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillText(total,cx,cy);
}
function drawThreatGauge(){
  const c=el('threatGauge');if(!c)return;c.width=120;c.height=70;
  const ctx=c.getContext('2d'),cx=60,cy=55;
  const total=sessionStats.total||1;
  const threat=sessionStats.fake/total;
  // Arc background
  ctx.beginPath();ctx.arc(cx,cy,40,-Math.PI,0);ctx.strokeStyle='rgba(255,255,255,0.06)';ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();
  // Arc fill
  ctx.beginPath();ctx.arc(cx,cy,40,-Math.PI,-Math.PI+Math.PI*threat);
  ctx.strokeStyle=threat>0.5?'#ff3366':threat>0.25?'#ffb700':'#00ffaa';ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();
  // Label
  const lvl=threat>0.6?'HIGH':threat>0.3?'MEDIUM':'LOW';
  el('threatLabel').textContent=lvl;
  el('threatLabel').style.color=threat>0.5?'var(--red)':threat>0.25?'var(--amber)':'var(--green)';
}

/* ═══ 5. EXPORT FORENSIC REPORT ═══ */
function exportReport(){
  const c=document.createElement('canvas');
  const W=800,H=500;c.width=W;c.height=H;
  const ctx=c.getContext('2d');
  // Background
  ctx.fillStyle='#03050a';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(0,210,255,0.15)';ctx.lineWidth=1;
  for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
  for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  // Header
  ctx.fillStyle='#00d2ff';ctx.font='bold 24px Syne,sans-serif';ctx.fillText('🛡️ VOICEGUARD FORENSIC REPORT',30,45);
  ctx.fillStyle='rgba(0,210,255,0.4)';ctx.font='11px JetBrains Mono';ctx.fillText(new Date().toISOString()+' · VoiceGuard v2.2',30,65);
  // Divider
  ctx.strokeStyle='rgba(0,210,255,0.2)';ctx.beginPath();ctx.moveTo(30,80);ctx.lineTo(W-30,80);ctx.stroke();
  // Get last result
  const h=analysisHistory[0];
  if(h){
    const iF=h.label==='FAKE';
    ctx.fillStyle=iF?'#ff3366':'#00ffaa';ctx.font='bold 48px Syne';ctx.fillText(h.label,30,140);
    ctx.fillStyle='#4a7090';ctx.font='13px JetBrains Mono';ctx.fillText(`Confidence: ${Math.round(h.confidence*100)}% · Features: ${h.features} · Time: ${h.processing_time_ms}ms`,30,170);
    ctx.fillText(`File: ${h.filename}`,30,190);
    ctx.fillStyle='#ddeeff';ctx.font='12px JetBrains Mono';
    const reason=h.verdict_reason||'';const words=reason.split(' ');let line='',ly=230;
    words.forEach(w=>{if(ctx.measureText(line+w).width>W-60){ctx.fillText(line,30,ly);ly+=18;line=''}line+=w+' '});
    ctx.fillText(line,30,ly);
    // Scores bar
    ly+=40;ctx.fillStyle='#4a7090';ctx.font='10px JetBrains Mono';ctx.fillText('PROBABILITY SCORES',30,ly);ly+=20;
    ctx.fillStyle='rgba(0,255,170,0.3)';ctx.fillRect(30,ly,Math.round(h.real_score*350),16);
    ctx.fillStyle='#00ffaa';ctx.font='11px JetBrains Mono';ctx.fillText(`Real: ${Math.round(h.real_score*100)}%`,390,ly+12);ly+=28;
    ctx.fillStyle='rgba(255,51,102,0.3)';ctx.fillRect(30,ly,Math.round(h.fake_score*350),16);
    ctx.fillStyle='#ff3366';ctx.fillText(`Fake: ${Math.round(h.fake_score*100)}%`,390,ly+12);ly+=40;
    // Features
    const tf=h.top_features||[];
    if(tf.length){
      ctx.fillStyle='#4a7090';ctx.font='10px JetBrains Mono';ctx.fillText('TOP FEATURES',30,ly);ly+=20;
      tf.forEach(f=>{
        ctx.fillStyle='rgba(0,210,255,0.2)';const bw=Math.round(f.weight/20*300);ctx.fillRect(30,ly,bw,14);
        ctx.fillStyle='#ddeeff';ctx.font='10px JetBrains Mono';ctx.fillText(`${f.name}: ${f.weight}%`,40,ly+11);ly+=20;
      });
    }
  }
  // Footer
  ctx.fillStyle='rgba(0,210,255,0.3)';ctx.font='9px JetBrains Mono';ctx.fillText('VOICEGUARD · AIML HACKATHON 2026 · MFCC · SPECTRAL · PROSODY · 3-MODEL ENSEMBLE',30,H-15);
  // Download
  const link=document.createElement('a');link.download='voiceguard_report_'+Date.now()+'.png';link.href=c.toDataURL('image/png');link.click();
}

/* ═══ 6. PARTICLES ═══ */
function initParticles(){
  const c=el('particleCanvas');if(!c)return;
  const ctx=c.getContext('2d');
  let W,H,particles=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight}
  resize();window.addEventListener('resize',resize);
  for(let i=0;i<60;i++)particles.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*2+0.5,o:Math.random()*.3+.1});
  function draw(){
    requestAnimationFrame(draw);ctx.clearRect(0,0,W,H);
    particles.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=`rgba(0,210,255,${p.o})`;ctx.fill();
    });
    // Connect nearby
    for(let i=0;i<particles.length;i++){
      for(let j=i+1;j<particles.length;j++){
        const dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<120){ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);ctx.lineTo(particles[j].x,particles[j].y);ctx.strokeStyle=`rgba(0,210,255,${.06*(1-d/120)})`;ctx.lineWidth=.5;ctx.stroke()}
      }
    }
  }
  draw();
}

/* ═══ 7. KEYBOARD SHORTCUTS ═══ */
function initKeyboard(){
  document.addEventListener('keydown',e=>{
    if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
    if(e.code==='Space'&&el('analyzeBtn')?.classList.contains('show')){e.preventDefault();analyze()}
    if(e.code==='KeyR'&&!e.ctrlKey&&!e.metaKey)resetAll();
    if(e.code==='KeyM')switchMode(currentMode==='mic'?'upload':'mic');
    if(e.code==='KeyE'&&analysisHistory.length)exportReport();
  });
}

/* ═══ REVEAL ANIMATIONS ═══ */
function initReveal(){
  const obs=new IntersectionObserver((entries)=>{entries.forEach(e=>{if(e.isIntersecting)e.target.classList.add('visible')})},{threshold:0.1});
  document.querySelectorAll('.reveal').forEach(el=>obs.observe(el));
}

/* ═══ 8. ANIMATED DNA (rotating) ═══ */
let dnaAnimFrame=null, dnaRotation=0, dnaData=null, dnaIsFake=false;
function drawDNAAnimated(){
  if(!dnaData)return;
  dnaRotation+=0.004;
  const c=el('dnaCanvas');if(!c)return;
  const W=220,H=220;c.width=W;c.height=H;
  const ctx=c.getContext('2d'),cx=W/2,cy=H/2;
  const feats=dnaData.top_features||[];
  const conf=dnaData.confidence||0.5;
  const col=dnaIsFake?[255,51,102]:[0,255,170];
  const pulse=0.95+0.05*Math.sin(Date.now()*0.003);

  // Outer ring with pulse
  ctx.beginPath();ctx.arc(cx,cy,95*pulse,0,Math.PI*2);
  ctx.strokeStyle=`rgba(${col},0.12)`;ctx.lineWidth=1;ctx.stroke();
  // Second outer ring
  ctx.beginPath();ctx.arc(cx,cy,100*pulse,0,Math.PI*2);
  ctx.strokeStyle=`rgba(${col},0.06)`;ctx.lineWidth=0.5;ctx.stroke();

  // Rotating DNA strands
  for(let s=0;s<4;s++){
    ctx.beginPath();
    const strands=48;
    for(let i=0;i<=strands;i++){
      const angle=(i/strands)*Math.PI*2-Math.PI/2+dnaRotation*(s%2===0?1:-0.7);
      const feat=feats[i%Math.max(feats.length,1)]||{weight:10};
      const baseR=25+s*16;
      const wobble=dnaIsFake?2+Math.sin(i*0.5)*1:6+Math.sin(i*1.7+s)*4;
      const r=(baseR+feat.weight*wobble/10)*pulse;
      const x=cx+Math.cos(angle)*r,y=cy+Math.sin(angle)*r;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.fillStyle=`rgba(${col},${0.02+s*0.015})`;ctx.fill();
    ctx.strokeStyle=`rgba(${col},${0.15+s*0.1})`;ctx.lineWidth=1.2;ctx.stroke();
  }

  // Orbiting data points
  feats.forEach((f,i)=>{
    const angle=(i/feats.length)*Math.PI*2-Math.PI/2+dnaRotation*1.5;
    const r=(35+f.weight*3)*pulse;
    const x=cx+Math.cos(angle)*r,y=cy+Math.sin(angle)*r;
    const glow=3+f.weight/6+Math.sin(Date.now()*0.005+i)*1;
    // Glow
    ctx.beginPath();ctx.arc(x,y,glow+4,0,Math.PI*2);
    ctx.fillStyle=`rgba(${col},0.08)`;ctx.fill();
    // Point
    ctx.beginPath();ctx.arc(x,y,glow,0,Math.PI*2);
    ctx.fillStyle=`rgba(${col},0.7)`;ctx.fill();
    // Line to center
    ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);
    ctx.strokeStyle=`rgba(${col},0.06)`;ctx.lineWidth=0.5;ctx.stroke();
  });

  // Center circle with glow
  const cPulse=22+2*Math.sin(Date.now()*0.004);
  ctx.beginPath();ctx.arc(cx,cy,cPulse+8,0,Math.PI*2);
  ctx.fillStyle=`rgba(${col},0.03)`;ctx.fill();
  ctx.beginPath();ctx.arc(cx,cy,cPulse,0,Math.PI*2);
  ctx.fillStyle=`rgba(${col},0.1)`;ctx.fill();
  ctx.strokeStyle=`rgba(${col},0.5)`;ctx.lineWidth=1.5;ctx.stroke();
  ctx.font='bold 14px JetBrains Mono';ctx.fillStyle=`rgb(${col})`;ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.fillText(Math.round(conf*100)+'%',cx,cy);

  dnaAnimFrame=requestAnimationFrame(drawDNAAnimated);
}
// Override drawDNA to start animation
const _origDrawDNA=drawDNA;
drawDNA=function(data,isFake){
  if(dnaAnimFrame)cancelAnimationFrame(dnaAnimFrame);
  dnaData=data;dnaIsFake=isFake;
  el('dnaLabel').textContent=isFake?'SYNTHETIC SIGNATURE — high symmetry detected':'ORGANIC SIGNATURE — natural asymmetry detected';
  drawDNAAnimated();
};

/* ═══ 9. SOUND EFFECTS (Web Audio synthesis) ═══ */
function playSound(type){
  try{
    const actx=new(window.AudioContext||window.webkitAudioContext)();
    const osc=actx.createOscillator(),gain=actx.createGain();
    osc.connect(gain);gain.connect(actx.destination);
    if(type==='success'){osc.frequency.setValueAtTime(523,actx.currentTime);osc.frequency.setValueAtTime(659,actx.currentTime+0.1);osc.frequency.setValueAtTime(784,actx.currentTime+0.2);gain.gain.setValueAtTime(0.08,actx.currentTime);gain.gain.exponentialRampToValueAtTime(0.001,actx.currentTime+0.4);osc.start();osc.stop(actx.currentTime+0.4)}
    else if(type==='warning'){osc.type='sawtooth';osc.frequency.setValueAtTime(220,actx.currentTime);osc.frequency.setValueAtTime(180,actx.currentTime+0.15);gain.gain.setValueAtTime(0.06,actx.currentTime);gain.gain.exponentialRampToValueAtTime(0.001,actx.currentTime+0.3);osc.start();osc.stop(actx.currentTime+0.3)}
    else if(type==='click'){osc.frequency.setValueAtTime(800,actx.currentTime);gain.gain.setValueAtTime(0.04,actx.currentTime);gain.gain.exponentialRampToValueAtTime(0.001,actx.currentTime+0.05);osc.start();osc.stop(actx.currentTime+0.05)}
  }catch(e){}
}

/* ═══ 10. REACTIVE PARTICLES ═══ */
let particleColor=[0,210,255];
function setParticleColor(r,g,b){particleColor=[r,g,b]}

// Patch initParticles to use reactive color
const _origInitParticles=initParticles;
initParticles=function(){
  const c=el('particleCanvas');if(!c)return;
  const ctx=c.getContext('2d');
  let W,H,particles=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight}
  resize();window.addEventListener('resize',resize);
  for(let i=0;i<60;i++)particles.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*2+0.5,o:Math.random()*.3+.1});
  function draw(){
    requestAnimationFrame(draw);ctx.clearRect(0,0,W,H);
    const[cr,cg,cb]=particleColor;
    particles.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0)p.x=W;if(p.x>W)p.x=0;if(p.y<0)p.y=H;if(p.y>H)p.y=0;
      ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=`rgba(${cr},${cg},${cb},${p.o})`;ctx.fill();
    });
    for(let i=0;i<particles.length;i++){
      for(let j=i+1;j<particles.length;j++){
        const dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<120){ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);ctx.lineTo(particles[j].x,particles[j].y);ctx.strokeStyle=`rgba(${cr},${cg},${cb},${.06*(1-d/120)})`;ctx.lineWidth=.5;ctx.stroke()}
      }
    }
  }
  draw();
};

/* ═══ 11. CONFIDENCE COUNTER ANIMATION ═══ */
function animateCounter(elId,target,duration=800){
  const elem=el(elId);if(!elem)return;
  const start=performance.now();const from=0;
  function tick(now){
    const t=Math.min((now-start)/duration,1);
    const eased=1-Math.pow(1-t,3); // ease-out cubic
    const val=Math.round(from+(target-from)*eased);
    elem.textContent=val+'%';
    if(t<1)requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ═══ 12. SCREEN FLASH ON VERDICT ═══ */
function flashScreen(isFake){
  const flash=document.createElement('div');
  flash.style.cssText=`position:fixed;inset:0;z-index:999;pointer-events:none;background:${isFake?'rgba(255,51,102,0.12)':'rgba(0,255,170,0.1)'};animation:flashFade 0.6s ease forwards`;
  document.body.appendChild(flash);
  setTimeout(()=>flash.remove(),700);
}
// Inject flash animation
if(!document.getElementById('flashStyle')){
  const s=document.createElement('style');s.id='flashStyle';
  s.textContent='@keyframes flashFade{0%{opacity:1}100%{opacity:0}}';
  document.head.appendChild(s);
}

/* ═══ PATCH showResult for new effects ═══ */
const _origShowResult=showResult;
showResult=function(data){
  const isFake=data.label==='FAKE';
  // Sound effect
  playSound(isFake?'warning':'success');
  // Flash
  flashScreen(isFake);
  // Reactive particles
  if(isFake)setParticleColor(255,51,102);else setParticleColor(0,255,170);
  // Reset particle color after 8s
  setTimeout(()=>setParticleColor(0,210,255),8000);
  // Call original
  _origShowResult(data);
  // Animate confidence counter
  animateCounter('confPct',Math.round(data.confidence*100));
  animateCounter('statConf',Math.round(data.confidence*100));
};
