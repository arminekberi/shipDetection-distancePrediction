'use strict';
const $ = id => document.getElementById(id);
const state = {frames:[], filtered:[], current:null, draft:[], background:false, chosen:'manual', dirty:false,
  scan:{}, queue:new URLSearchParams(location.search).get('queue')==='yolo_background'?'yolo_background':'invalid', filterStatus:new URLSearchParams(location.search).get('queue')==='yolo_background'?'all':'pending', query:'', limit:100, selected:-1, tool:'select', zoom:1, undo:[], busy:false, loading:false, loadToken:0};
const queues = [['yolo_background','Корабля нет, YOLO нашёл'],['invalid','Некорректные рамки'],['conflict','Разные версии'],['names','Совпадения имён'],['all','Все кадры']];
const statuses = {pending:'Не проверен',approved:'Подтверждён',excluded:'Исключён',deferred:'Отложен'};
const flagNames = {invalid:'Некорректная рамка',conflict:'Разметка различается',names:'Имя есть в train и val'};
const reasonNames = {recording_identity_unresolved:'Запись не установлена',reserved_evaluation_session:'Запись отложена для оценки',mono_or_low_color_review:'Моно или мало цвета'};
const clone = x => JSON.parse(JSON.stringify(x));
const clamp = x => Math.max(0, Math.min(1, x));
const imageURL = id => '/api/review/image?id='+encodeURIComponent(id);
const node = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
function notify(message, error=false){$('notice').textContent=message;$('notice').classList.toggle('error',error);$('notice').hidden=false;}
async function api(path, data){
 const response=await fetch('/api/review/'+path, data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 const body=await response.json();if(!response.ok)throw new Error(body.error||'Не удалось выполнить запрос.');return body;
}
function queueMatch(frame,queue=state.queue){return queue==='all'||(queue==='yolo_background'?frame.yolo_background:frame.flags.includes(queue));}
function filterFrames(){
 const query=state.query.trim().toLowerCase(),status=state.filterStatus;
 state.filtered=state.frames.filter(f=>queueMatch(f)&&(status==='all'||f.status===status)&&(!query||(f.id+' '+f.session).toLowerCase().includes(query)));
}
function renderIndex(){
 $('statusFilter').value=state.filterStatus;
 $('scanStatus').hidden=state.queue!=='yolo_background';
 $('scanStatus').textContent=state.scan.model?`YOLO: ${state.scan.model.weights.split('/').pop()} · conf ≥ ${state.scan.model.conf} · проверено ${state.scan.scanned}/${state.scan.total_background}${state.scan.complete?'':' · неполный проход'}. Обновите страницу после сканирования.`:'Сначала запустите: venv/bin/python review_background.py, затем обновите страницу.';
 const done=state.frames.filter(f=>['approved','excluded'].includes(f.status)).length;
 $('progressText').textContent=`${done.toLocaleString('ru')} / ${state.frames.length.toLocaleString('ru')} проверено`;
 $('progress').max=state.frames.length;$('progress').value=done;
 $('queues').replaceChildren(...queues.map(([id,title])=>{
  const b=node('button',undefined,'queue');b.setAttribute('aria-pressed',String(state.queue===id));
  b.append(node('span',title),node('span',state.frames.filter(f=>queueMatch(f,id)).length));
  b.onclick=()=>guard(async()=>{state.queue=id;if(id==='yolo_background')state.filterStatus='all';state.limit=100;filterFrames();renderIndex();await loadFirst();});return b;
 }));
 filterFrames();if(state.queue==='yolo_background')state.filtered.sort((a,b)=>b.yolo_confidence-a.yolo_confidence);$('listCount').textContent=`Кадров: ${state.filtered.length}`;
 $('frameList').replaceChildren(...state.filtered.slice(0,state.limit).map(f=>{
  const b=node('button',f.stem,'frame-item'+(state.current?.id===f.id?' active':''));
  b.append(node('small',f.split+' · '+statuses[f.status]+(state.queue==='yolo_background'?` · YOLO ${(f.yolo_confidence*100).toFixed(1)}%`:'')));b.onclick=()=>guard(()=>loadFrame(f.id));return b;
 }));
 $('more').hidden=state.limit>=state.filtered.length;updateNavigation();
}
function updateNavigation(){
 const index=state.filtered.findIndex(f=>f.id===state.current?.id);
 $('previous').disabled=state.busy||state.loading||index<=0;
 $('next').disabled=state.busy||state.loading||index<0||index>=state.filtered.length-1;
 if(state.current)$('framePosition').textContent=index>=0?`${index+1} из ${state.filtered.length} в очереди · ${state.current.session||'Запись неизвестна'}`:'Кадр вне текущего фильтра';
}
let pendingLeave=null;
function guard(action){
 if(state.busy)return;
 if(!state.dirty)return Promise.resolve(action()).catch(e=>notify(e.message,true));
 pendingLeave=action;if(!$('leaveDialog').open)$('leaveDialog').showModal();
}
$('stay').onclick=()=>{$('leaveDialog').close();pendingLeave=null;};
$('discard').onclick=()=>{const action=pendingLeave;pendingLeave=null;$('leaveDialog').close();state.dirty=false;Promise.resolve(action?.()).catch(e=>notify(e.message,true));};
async function loadFirst(){if(state.filtered.length)await loadFrame(state.filtered[0].id);else empty();}
function empty(){
 ++state.loadToken;state.current=null;state.loading=false;state.dirty=false;state.draft=[];
 $('stage').hidden=true;$('emptyState').hidden=false;$('emptyState').textContent='В этой очереди больше нет кадров. Выберите другую очередь или измените фильтр.';
 $('frameName').textContent='Очередь пуста';$('splitBadge').textContent='—';$('framePosition').textContent='';
 for(const id of ['versions','flags','history','peers'])$(id).replaceChildren();
 $('yoloSummary').hidden=true;$('peerSection').hidden=true;$('recording').value='';$('note').value='';updateControls();renderIndex();
}
async function loadFrame(id){
 const token=++state.loadToken;state.loading=true;state.current=null;state.dirty=false;state.draft=[];
 $('stage').hidden=true;$('emptyState').hidden=false;$('emptyState').textContent='Загружаю кадр…';updateControls();
 try{
  const detail=await api('frame?id='+encodeURIComponent(id));if(token!==state.loadToken)return;
  const image=new Image();image.src=imageURL(id);await image.decode();if(token!==state.loadToken)return;
  state.current=detail;state.loading=false;state.selected=-1;state.undo=[];state.zoom=1;setTool('select');
  const indexed=state.frames.find(f=>f.id===detail.id);
  if(indexed){indexed.status=detail.decision?.status||'pending';indexed.version=detail.decision?.version||0;indexed.yolo_background=!!detail.yolo_prediction?.detections.length;}
  const decision=detail.decision,source=detail.versions[0];
  state.draft=clone(decision?decision.boxes:source.invalid_lines.length?[]:source.boxes);
  state.background=decision?.background_confirmed||false;
  state.chosen=decision?.chosen_version||(source.invalid_lines.length?'manual':'source');
  $('note').value=decision?.note||'';$('recording').value=decision?.recording||'';
  $('frameImage').src=image.src;$('frameImage').alt=`${detail.split} · ${detail.stem}`;
  $('stage').style.width='100%';$('zoomLabel').textContent='100%';$('viewport').scrollTo(0,0);
  $('stage').hidden=false;$('emptyState').hidden=true;
  $('frameName').textContent=detail.stem;$('splitBadge').textContent=detail.split;
  const flags=[...detail.flags.map(f=>flagNames[f]),...detail.reasons.filter(r=>reasonNames[r]).map(r=>reasonNames[r])];
  $('flags').replaceChildren(...flags.map(f=>node('span',f,'flag')));
  $('yoloSummary').hidden=!detail.yolo_prediction;
  $('yoloSummary').textContent=detail.yolo_prediction?`Жёлтые рамки — YOLO (${detail.yolo_model.weights.split('/').pop()}). Это подсказки для проверки. Если корабль есть, обведите его; если нет — подтвердите фон ещё раз.`:'';
  renderVersions();renderPeers();renderHistory();renderIndex();renderBoxes();updateControls();
 }catch(e){if(token===state.loadToken){state.loading=false;state.current=null;$('emptyState').textContent=e.message;updateControls();notify(e.message,true);}}
}
function renderVersions(){
 $('versions').replaceChildren(...state.current.versions.map((v,i)=>{
  const card=node('div',undefined,'version-card'),b=node('button',v.name);
  b.disabled=!!v.unavailable||!!v.invalid_lines?.length;
  b.onclick=()=>{pushUndo();state.draft=clone(v.boxes);state.background=false;state.chosen=v.key;state.selected=-1;changed();};
  card.append(b);
  if(v.unavailable)card.append(node('small',v.error));
  else {card.append(node('p',v.boxes.length?`Рамок: ${v.boxes.length}`:'Пустая разметка'));
   if(v.invalid_lines.length)card.append(node('small','Есть некорректные координаты. Нарисуйте рамку по изображению.'));
  }
  card.style.borderLeftColor=i===0?'var(--red)':'var(--accent)';card.style.borderLeftWidth='3px';return card;
 }));
}
function renderPeers(){
 const peers=state.current.peers||[];$('peerSection').hidden=!peers.length;
 $('peers').replaceChildren(...peers.map(id=>{
  const b=node('button',undefined,'peer'),img=node('img');img.src=imageURL(id);img.alt=id;img.loading='lazy';
  b.append(img,node('span',id));b.onclick=()=>guard(async()=>{
   state.queue='all';state.filterStatus='all';state.query=state.current.stem;$('statusFilter').value=state.filterStatus;$('search').value=state.query;renderIndex();await loadFrame(id);
  });return b;
 }));
}
function renderHistory(){
 $('history').replaceChildren(...(state.current.history.length?state.current.history.map(d=>node('p',`№${d.version} · ${statuses[d.status]} · ${new Date(d.updated_at).toLocaleString('ru')}${d.note?' — '+d.note:''}`)):[node('p','Решений пока нет.')]));
}
function snapshotDraft(){return {boxes:clone(state.draft),background:state.background,chosen:state.chosen};}
function pushUndo(){state.undo.push(snapshotDraft());if(state.undo.length>50)state.undo.shift();}
function changed(){state.dirty=true;renderBoxes();updateControls();}
function validBoxes(){return state.draft.every(b=>b.every(Number.isFinite)&&b[0]>=0&&b[1]>=0&&b[2]<=1&&b[3]<=1&&b[0]<b[2]&&b[1]<b[3]);}
function updateControls(){
 const ready=!!state.current&&!state.busy&&!state.loading;
 $('save').disabled=!ready||!validBoxes()||(!state.draft.length&&!state.background);
 for(const id of ['defer','exclude','background','drawTool','selectTool','note','recording','zoomIn','zoomOut','reload'])$(id).disabled=!ready;
 $('undo').disabled=!ready||!state.undo.length;$('deleteBox').disabled=!ready||state.selected<0;
 $('savedState').classList.toggle('unsaved',state.dirty);
 $('savedState').textContent=state.loading?'Загрузка…':!state.current?'Выберите кадр':state.dirty?'Есть несохранённые изменения':state.current.decision?`${statuses[state.current.decision.status]} · сохранено`:'Решение ещё не сохранено';
 $('boxCount').textContent=state.draft.length?`Кораблей в решении: ${state.draft.length}`:state.background?'Подтверждено: кораблей нет':'Нет рамок. Нарисуйте корабль или нажмите «Корабля нет».';
 $('background').setAttribute('aria-pressed',String(state.background));updateNavigation();
 document.querySelectorAll('#versions button').forEach((b,i)=>{const v=state.current?.versions[i];b.disabled=!ready||!!v?.unavailable||!!v?.invalid_lines?.length;});
}
const NS='http://www.w3.org/2000/svg';
function svgNode(tag,attrs){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);return n;}
function boxRect(box,color,extra={}){return svgNode('rect',{x:box[0]*1000,y:box[1]*1000,width:Math.max(0,box[2]-box[0])*1000,height:Math.max(0,box[3]-box[1])*1000,fill:'none',stroke:color,'stroke-width':2,'vector-effect':'non-scaling-stroke',...extra});}
function renderBoxes(){
 const svg=$('overlay');svg.replaceChildren();if(!state.current)return;
 if($('showReferences').checked){state.current.versions.forEach((v,i)=>{for(const box of v.boxes||[])svg.append(boxRect(box,i===0?'#ff8c87':'#53d9ea',{'stroke-dasharray':i===0?'6 4':'2 4','pointer-events':'none'}));});}
 if($('showYolo').checked){for(const detection of state.current.yolo_prediction?.detections||[]){
  svg.append(boxRect(detection.box,'#ffd166',{'stroke-width':3,'pointer-events':'none'}));
  const label=svgNode('text',{x:detection.box[0]*1000,y:Math.max(18,detection.box[1]*1000-7),fill:'#ffd166',stroke:'#080e16','stroke-width':3,'paint-order':'stroke','font-size':18,'pointer-events':'none'});
  label.textContent=`YOLO ${(detection.confidence*100).toFixed(1)}%`;svg.append(label);
 }}
 state.draft.forEach((box,i)=>{
  svg.append(boxRect(box,'#8beaab',{'data-index':i,fill:i===state.selected?'#8beaab20':'#8beaab08','stroke-width':i===state.selected?3:2,style:'cursor:move'}));
  if(i===state.selected){const bounds=$('stage').getBoundingClientRect(),dx=5/bounds.width*1000,dy=5/bounds.height*1000;
   [[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]].forEach(([x,y],corner)=>{
    svg.append(svgNode('rect',{x:x*1000-dx,y:y*1000-dy,width:2*dx,height:2*dy,fill:'#8beaab',stroke:'#072219','stroke-width':1,'vector-effect':'non-scaling-stroke','data-index':i,'data-corner':corner,style:`cursor:${corner%2?'nesw':'nwse'}-resize`}));
   });
  }
 });
}
function setTool(tool){state.tool=tool;$('stage').classList.toggle('drawing',tool==='draw');$('selectTool').setAttribute('aria-pressed',String(tool==='select'));$('drawTool').setAttribute('aria-pressed',String(tool==='draw'));}
function point(e){const r=$('stage').getBoundingClientRect();return [clamp((e.clientX-r.left)/r.width),clamp((e.clientY-r.top)/r.height)];}
let gesture=null;
$('overlay').addEventListener('pointerdown',e=>{
 if(!state.current||state.busy||e.button!==0)return;
 e.preventDefault();const p=point(e),index=e.target.getAttribute('data-index');
 gesture={start:p,before:snapshotDraft(),moved:false,pointer:e.pointerId};
 if(state.tool==='draw'){gesture.kind='draw';gesture.index=state.draft.length;state.draft.push([...p,...p]);state.selected=gesture.index;}
 else if(index!==null){state.selected=Number(index);gesture.index=state.selected;gesture.box=[...state.draft[state.selected]];const corner=e.target.getAttribute('data-corner');gesture.kind=corner===null?'move':'resize';gesture.corner=Number(corner);}
 else {state.selected=-1;gesture=null;renderBoxes();updateControls();return;}
 $('overlay').setPointerCapture(e.pointerId);renderBoxes();updateControls();
});
$('overlay').addEventListener('pointermove',e=>{
 if(!gesture||e.pointerId!==gesture.pointer)return;const p=point(e),g=gesture;g.moved=true;
 if(g.kind==='draw')state.draft[g.index]=[Math.min(g.start[0],p[0]),Math.min(g.start[1],p[1]),Math.max(g.start[0],p[0]),Math.max(g.start[1],p[1])];
 else if(g.kind==='move'){
  const b=g.box,dx=Math.max(-b[0],Math.min(1-b[2],p[0]-g.start[0])),dy=Math.max(-b[1],Math.min(1-b[3],p[1]-g.start[1]));
  state.draft[g.index]=[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy];
 }else{const b=g.box,opposite=[[b[2],b[3]],[b[0],b[3]],[b[0],b[1]],[b[2],b[1]]][g.corner];state.draft[g.index]=[Math.min(p[0],opposite[0]),Math.min(p[1],opposite[1]),Math.max(p[0],opposite[0]),Math.max(p[1],opposite[1])];}
 renderBoxes();
});
function finishGesture(cancel=false){
 if(!gesture)return;const g=gesture;gesture=null;
 const b=state.draft[g.index],img=$('frameImage');
 if(!cancel&&!g.moved&&g.kind!=='draw'){renderBoxes();updateControls();return;}
 if(cancel||!g.moved||(b[2]-b[0])*img.naturalWidth<2||(b[3]-b[1])*img.naturalHeight<2){state.draft=g.before.boxes;state.selected=-1;renderBoxes();updateControls();return;}
 state.undo.push(g.before);state.chosen='manual';state.background=false;setTool('select');changed();
}
$('overlay').addEventListener('pointerup',()=>finishGesture());$('overlay').addEventListener('pointercancel',()=>finishGesture(true));
$('drawTool').onclick=()=>setTool('draw');$('selectTool').onclick=()=>setTool('select');
$('deleteBox').onclick=()=>{if(state.selected<0)return;pushUndo();state.draft.splice(state.selected,1);state.selected=-1;state.chosen='manual';state.background=false;changed();};
$('undo').onclick=()=>{const prev=state.undo.pop();if(!prev)return;state.draft=prev.boxes;state.background=prev.background;state.chosen=prev.chosen;state.selected=-1;changed();};
$('background').onclick=()=>{pushUndo();state.draft=[];state.background=true;state.chosen='manual';state.selected=-1;changed();};
$('note').oninput=$('recording').oninput=()=>{state.dirty=true;updateControls();};
$('showReferences').onchange=renderBoxes;$('showYolo').onchange=renderBoxes;
function zoom(delta){state.zoom=Math.max(1,Math.min(4,state.zoom+delta));$('stage').style.width=state.zoom*100+'%';$('zoomLabel').textContent=state.zoom*100+'%';renderBoxes();}
$('zoomIn').onclick=()=>zoom(.5);$('zoomOut').onclick=()=>zoom(-.5);
window.addEventListener('resize',renderBoxes);
async function save(status){
 if(!state.current||state.busy||state.loading)return;
 if(status==='approved'&&$('save').disabled)return;
 finishGesture(true);
 const id=state.current.id,oldList=state.filtered.map(f=>f.id),position=oldList.indexOf(id);
 state.busy=true;updateControls();
 try{
  const decision=await api('save',{id,version:state.current.decision?.version||0,status,boxes:state.draft,
   background_confirmed:state.background,chosen_version:state.chosen,note:$('note').value,recording:$('recording').value});
  state.dirty=false;const entry=state.frames.find(f=>f.id===id);entry.status=decision.status;entry.version=decision.version;entry.yolo_background=false;state.current.yolo_prediction=null;$('yoloSummary').hidden=true;renderBoxes();
  state.current.decision=decision;state.current.history.unshift(decision);renderIndex();
  notify(`${statuses[status]}: ${id}. Решение сохранено.`);
  const after=oldList.slice(position+1).find(fid=>state.filtered.some(f=>f.id===fid));
  const next=after||state.filtered.find(f=>f.id!==id&&(state.queue==='yolo_background'||f.status==='pending'))?.id;
  if(next)await loadFrame(next);else if(!state.filtered.length)empty();else{renderHistory();}
 }catch(e){notify(e.message,true);}finally{state.busy=false;updateControls();}
}
$('save').onclick=()=>save('approved');$('defer').onclick=()=>save('deferred');$('exclude').onclick=()=>save('excluded');
function navigate(offset){const i=state.filtered.findIndex(f=>f.id===state.current?.id),target=state.filtered[i+offset];if(target)guard(()=>loadFrame(target.id));}
$('previous').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);
$('reload').onclick=()=>guard(()=>loadFrame(state.current.id));
$('statusFilter').onchange=()=>{const value=$('statusFilter').value;$('statusFilter').value=state.filterStatus;guard(async()=>{state.filterStatus=value;$('statusFilter').value=value;state.limit=100;renderIndex();await loadFirst();});};
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{const value=$('search').value;if(state.dirty)$('search').value=state.query;guard(async()=>{state.query=value;$('search').value=value;state.limit=100;renderIndex();await loadFirst();});},350);};
$('more').onclick=()=>{state.limit+=100;renderIndex();};
$('export').onclick=async()=>{
 if(state.dirty){notify('Сначала сохраните решение текущего кадра. В архив входят только сохранённые решения.',true);return;}
 $('export').disabled=true;
 try{const r=await fetch('/api/review/export');if(!r.ok){const e=await r.json();throw new Error(e.error);}
  const blob=await r.blob(),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download='annotation-review.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  notify('Архив решений скачан. В нём проверенные метки и история происхождения; подготовка train/val/test — отдельный шаг.');
 }catch(e){notify(e.message,true);}finally{$('export').disabled=false;}
};
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('keydown',e=>{
 if($('leaveDialog').open||e.target.matches('input,textarea,select')||!state.current||state.busy||state.loading||e.ctrlKey||e.metaKey||e.altKey)return;
 if(e.key==='ArrowRight'){e.preventDefault();navigate(1);}else if(e.key==='ArrowLeft'){e.preventDefault();navigate(-1);}
 else if(e.key==='Enter'&&!e.target.matches('button')){e.preventDefault();save('approved');}
 else if(e.key.toLowerCase()==='d'||e.key.toLowerCase()==='в'){e.preventDefault();setTool('draw');}
 else if(e.key.toLowerCase()==='n'||e.key.toLowerCase()==='т'){e.preventDefault();$('background').click();}
 else if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();$('deleteBox').click();}
 else if(e.key==='Escape'){finishGesture(true);setTool('select');}
});
(async()=>{try{const data=await api('index');state.frames=data.frames;state.scan=data.background_scan||{};$('export').disabled=false;renderIndex();await loadFirst();}catch(e){$('emptyState').textContent=e.message;notify(e.message,true);}})();
