(() => {
  const videos = [...document.querySelectorAll('.candidate video')];
  const play = document.querySelector('#grid-play');
  const restart = document.querySelector('#grid-reset');
  const slider = document.querySelector('#grid-time');
  const timeOutput = document.querySelector('#grid-clock');
  const message = document.querySelector('#grid-message');
  const limit = Number(slider.max);
  let loaded, running = false, time = 0, last = 0, frame = 0;
  function clockLabel() { slider.value = String(time); timeOutput.value = `${time.toFixed(1)} / ${limit.toFixed(1)} s`; }
  function stop() { running = false; cancelAnimationFrame(frame); videos.forEach(v => v.pause()); play.textContent = 'Play six recordings'; play.setAttribute('aria-pressed', 'false'); }
  function loadVideos() {
    if (!loaded) loaded = Promise.all(videos.map(video => new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('A recording is taking longer to load. Try again, or open its individual replay.')), 30000);
      const ready = () => {clearTimeout(timeout); resolve();};
      video.addEventListener('loadeddata', ready, {once:true});
      video.addEventListener('error', () => {clearTimeout(timeout);reject(new Error('A recording could not be loaded. Its individual replay is linked below.'));}, {once:true});
      video.src = video.dataset.src;video.load();
    }))).catch(error => { loaded = null;throw error; });
    return loaded;
  }
  function align() { videos.forEach(v => {if(Number.isFinite(v.duration)) {const target = Math.min(time, Math.max(0,v.duration-.04));if(Math.abs(v.currentTime-target)>.16)v.currentTime=target;}}); }
  function tick(now) {
    if(!running)return;
    time=Math.min(limit,time+Math.min((now-last)/1000,.1));last=now;align();clockLabel();
    if(time>=limit){stop();return;}frame=requestAnimationFrame(tick);
  }
  play.addEventListener('click',async()=>{
    if(running){stop();return;}
    play.disabled=true;message.textContent='Loading the six recorded trials…';
    try {
      await loadVideos();if(time>=limit)time=0;align();
      await Promise.all(videos.map(v=>v.play()));
      running=true;last=performance.now();play.textContent='Pause recordings';play.setAttribute('aria-pressed','true');message.textContent='';frame=requestAnimationFrame(tick);
    }catch(e){stop();message.textContent=e.message;}finally{play.disabled=false;}
  });
  slider.addEventListener('input',()=>{time=Number(slider.value);clockLabel();if(loaded)align();});
  slider.addEventListener('change',async()=>{stop();try{await loadVideos();align();message.textContent='';}catch(e){message.textContent=e.message;}});
  restart.addEventListener('click',()=>{stop();time=0;align();clockLabel();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&running)stop();},{threshold:0}).observe(document.querySelector('#candidate-grid'));
  clockLabel();

  const selector=document.querySelector('#design-select');
  const designRows=document.querySelector('#design-rows');
  const detailTitle=document.querySelector('#design-title');
  const detailText=document.querySelector('#design-text');
  const designLink=document.querySelector('#design-replay');
  const explanations={
    nominal:['The reference actuator.','The nominal Atlas drive did not complete this course. Its 9:1 gearing, 57.3 rpm/V winding and current limit provide the reference for every design comparison here.'],
    kv_high:['A faster winding. A different outcome.','This winding raises Kv by 50%. The model changes Kt, resistance and inductance together. The recorded robot completes the course with a lower ideal stall torque than the nominal design.'],
    gear_low:['Less reduction. More joint speed.','The 6:1 transmission trades ideal output torque for speed. The same frozen policy completes the course in this trial. Native joint armature is retained in these course recordings.'],
    mass_added_double:['Six kilograms change the movement.','The added mass is applied to actuator carrier bodies, including their inertia. This run ends after the staircase. This is an added-mass experiment; the motors were not relocated.'],
    torque_low:['A lower current limit still completes.','The peak-current limit falls from 55 A to 36.85 A. This candidate completes the course in the recorded trial, showing why stall torque alone cannot rank task performance.'],
    torque_focused:['A combined design change.','This candidate combines lower Kv, higher gearing and 1.2 kg of added carrier mass. It completes the challenge. The outcome reflects the combined intervention; it does not isolate any one parameter.']
  };
  let candidates;
  function selectDesign(id, scroll=false){
    if(!candidates)return;
    const c=candidates.variants.find(v=>v.id===id), baseline=candidates.variants.find(v=>v.id==='nominal');
    if(!c?.specifications)return;
    selector.value=id;document.querySelectorAll('[data-candidate]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.candidate===id));b.closest('.candidate').classList.toggle('active',b.dataset.candidate===id);});
    const [title,description]=explanations[id];detailTitle.textContent=title;detailText.textContent=description;
    designLink.href=`lab/?design=${encodeURIComponent(id)}#terrain`;
    document.querySelector('#design-outcome').textContent=c.passed?`Completed · ${c.duration.toFixed(2)} s`:`Not completed · ${c.duration.toFixed(2)} s`;
    const fields=[['Kv · phase-peak rpm/V','Kv_phase_peak_rpm_per_V',1],['Gear ratio','gear_ratio',1],['Peak current · A','peak_current_A',2],['Phase resistance · mΩ','phase_R_ohm',1,1000],['Ideal stall torque · N·m','nominal_stall_peak_joint_Nm',1],['Added carrier mass · kg',null,1]];
    designRows.replaceChildren(...fields.map(([label,key,digits,multiplier=1])=>{
      const a=key?baseline.specifications[key]*multiplier:baseline.parameters.mass,b=key?c.specifications[key]*multiplier:c.parameters.mass;
      const tr=document.createElement('tr');[label,a.toFixed(digits),b.toFixed(digits)].forEach((text,i)=>{const td=document.createElement('td');td.textContent=text;if(i===2&&Math.abs(a-b)>1e-6)td.className='changed';tr.append(td);});return tr;
    }));
    if(scroll)document.querySelector('#designs').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  }
  fetch('assets/research/data/candidates.json').then(r=>{if(!r.ok)throw Error('Design data unavailable');return r.json();}).then(data=>{candidates=data;selectDesign('kv_high');}).catch(()=>{document.querySelector('#design-text').textContent='The comparison data could not load. Open the recorded designs in the interactive lab.';});
  selector.addEventListener('change',()=>selectDesign(selector.value));
  document.querySelectorAll('[data-candidate]').forEach(b=>b.addEventListener('click',()=>selectDesign(b.dataset.candidate,true)));

  const descriptions={terrain:'A 19 m course with stairs, a gap, narrow crossing, rubble, slope and physical push. Select a recorded design, orbit the body-state replay and inspect the matching video.',pen:'Three paired training seeds, synchronized comparison videos and 3D replays. Inspect motion, current, torque and the complete test outcomes.',g1:'Recorded G1 walking and physical actuator-mass comparisons. This is the walking experiment.',anymal:'ANYmal locomotion with recorded actuator designs, matched videos and joint telemetry.',transfer:'Two arms performing a continuous transfer sequence, with actuator comparisons and object motion.',allegro:'Allegro cube reorientation and sustained-hold trials, including failed outcomes.',rubik:'Wuji hands with a validated free-cube quarter-turn, fixture trials and a failed longer attempt. Full cube solving remains unfinished.',assembly:'Peg insertion with and without a physical side push, plus the recorded gear-transfer trial.'};
  const labels={terrain:'ANYmal obstacle course',pen:'Sharpa pen spinning',g1:'G1 walking',anymal:'ANYmal walking',transfer:'Two-arm transfer',allegro:'Allegro hand',rubik:'Wuji cube',assembly:'Precision assembly'};
  let task='terrain',viewerLoaded=false;
  function loadViewer(){
    if(!viewerLoaded){const f=document.createElement('iframe');f.id='demo-frame';f.title=`${labels[task]} — synchronized video and 3D replay`;f.allow='fullscreen';f.setAttribute('allowfullscreen','');f.src=`lab/#${task}`;document.querySelector('#viewer-placeholder').replaceWith(f);viewerLoaded=true;}
    else {const f=document.querySelector('#demo-frame');f.src=`lab/#${task}`;f.title=`${labels[task]} — synchronized video and 3D replay`;}
  }
  document.querySelector('#load-viewer').addEventListener('click',loadViewer);
  document.querySelectorAll('[data-demo]').forEach(b=>b.addEventListener('click',()=>{
    task=b.dataset.demo;document.querySelectorAll('[data-demo]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));
    document.querySelector('#demo-description').textContent=descriptions[task];document.querySelector('#demo-full').href=`lab/#${task}`;
    if(viewerLoaded)loadViewer();else {document.querySelector('#viewer-title').textContent=labels[task];document.querySelector('#viewer-description').textContent=descriptions[task];}
  }));
  // Preserve direct links to the old interactive section.
  if(location.hash==='#demo-update')loadViewer();
})();
