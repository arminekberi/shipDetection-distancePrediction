// Exercise the review queue and prediction overlay without a browser or real decisions.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function element() {
  return {children:[], attributes:{}, checked:true, style:{}, classList:{toggle(){}},
    addEventListener(){}, setAttribute(k,v){this.attributes[k]=v;},
    replaceChildren(...children){this.children=children;}, append(...children){this.children.push(...children);}};
}
const nodes = new Map();
const document = {
  getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);},
  createElement:element, createElementNS:element, querySelectorAll(){return [];}
};
const context = vm.createContext({document, window:{addEventListener(){}},
  location:{search:'?queue=yolo_background'}, URLSearchParams,
  fetch:()=>new Promise(()=>{}), setTimeout, clearTimeout});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../viewer/dataset_review.js'),'utf8')+
  '\nglobalThis.review={state,renderIndex,renderBoxes};',context);
const {state,renderIndex,renderBoxes}=context.review;
assert.equal(state.filterStatus,'all');
state.frames=[
  {id:'train/low',stem:'low',split:'train',status:'approved',flags:[],yolo_background:true,yolo_confidence:.2},
  {id:'train/high',stem:'high',split:'train',status:'approved',flags:[],yolo_background:true,yolo_confidence:.9},
  {id:'val/other',stem:'other',split:'val',status:'pending',flags:[],yolo_background:false,yolo_confidence:0}
];
renderIndex();
assert.deepEqual(Array.from(state.filtered,f=>f.id),['train/high','train/low']);
assert.equal(document.getElementById('statusFilter').value,'all');
assert.match(document.getElementById('scanStatus').textContent,/review_background.py/);
state.current={versions:[],yolo_prediction:{detections:[{box:[.1,.2,.3,.4],confidence:.9}]}};
renderBoxes();
assert.equal(document.getElementById('overlay').children.length,2);
assert.equal(document.getElementById('overlay').children[1].textContent,'YOLO 90.0%');
assert.equal(state.draft.length,0, 'Suggestions must not change the saved background draft');
document.getElementById('showYolo').checked=false;
renderBoxes();
assert.equal(document.getElementById('overlay').children.length,0);
state.current=null;state.frames[1].yolo_background=false;
renderIndex();
assert.deepEqual(Array.from(state.filtered,f=>f.id),['train/low']);
console.log('Dataset review queue and YOLO overlay checks passed.');
