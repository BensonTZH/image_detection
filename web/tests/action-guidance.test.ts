import test from 'node:test';
import assert from 'node:assert/strict';
import { correctionResolved, updateActionInstruction, type ActionInstructionState } from '../lib/action-guidance.ts';
import { emptyInstructionState, emptyCaptureHold, updateCaptureHold } from '../lib/live-guidance.ts';
import { DEFAULT_ALIGNMENT_REFERENCE, type AlignmentMeasurements } from '../lib/pose.ts';

const evidence={sameTrack:true,fresh:true,reliable:true,resolved:false,sideConfirmed:false,passingFrame:false,maxGapMs:1200};
function confirmed(): ActionInstructionState {
  let state:ActionInstructionState=emptyInstructionState();
  for(const now of [0,400,800]) state=updateActionInstruction(state,'step_left',now,evidence);
  assert.equal(state.guidance,'step_left');return state;
}

test('confirmed movement stays constant through unresolved readings instead of expiring every 1.2 seconds',()=>{
  let state=confirmed(),hold=emptyCaptureHold();
  for(let now=1200;now<=8000;now+=400) {
    state=updateActionInstruction(state,'steady',now,evidence);
    hold=updateCaptureHold(hold,now,false);
    assert.equal(state.guidance,'step_left');assert.equal(hold.ready,false);
  }
  assert.equal(state.lastSupportedAt,800,'uncertainty is not new support');
});

test('one resolved reading does not erase the action, two consecutive ones do',()=>{
  let state=confirmed();
  state=updateActionInstruction(state,'steady',1200,{...evidence,resolved:true});
  assert.equal(state.guidance,'step_left');
  state=updateActionInstruction(state,'steady',1600,evidence);
  state=updateActionInstruction(state,'steady',2000,{...evidence,resolved:true});
  assert.equal(state.guidance,'step_left');
  state=updateActionInstruction(state,'steady',2400,{...evidence,resolved:true});
  assert.equal(state.guidance,'steady');assert.notEqual(state.guidance,'aligned');
});

test('replacement action is confirmed without an intervening neutral instruction',()=>{
  let state=confirmed();
  for(let now=1200;now<=3200;now+=400)state=updateActionInstruction(state,'steady',now,evidence);
  state=updateActionInstruction(state,'lower_phone',3600,evidence);assert.equal(state.guidance,'step_left');
  state=updateActionInstruction(state,'lower_phone',4000,evidence);assert.equal(state.guidance,'step_left');
  state=updateActionInstruction(state,'lower_phone',4400,evidence);assert.equal(state.guidance,'lower_phone');
});

test('brief weak corners preserve a direction but prolonged uncertainty, loss and new tracks clear it',()=>{
  let state=confirmed();
  for(let now=1200;now<=3200;now+=400)state=updateActionInstruction(state,'steady',now,evidence);
  state=updateActionInstruction(state,'steady',3600,{...evidence,reliable:false});assert.equal(state.guidance,'step_left');
  state=updateActionInstruction(state,'steady',4000,{...evidence,reliable:false});assert.equal(state.guidance,'step_left');
  state=updateActionInstruction(state,'steady',4800,{...evidence,reliable:false});assert.equal(state.guidance,'steady');
  assert.equal(updateActionInstruction(confirmed(),'searching',1200,{...evidence,reliable:false}).guidance,'searching');
  assert.equal(updateActionInstruction(confirmed(),'steady',1200,{...evidence,sameTrack:false}).guidance,'steady');
});

test('stale measurements cannot resolve or replace an instruction',()=>{
  let state=confirmed();
  state=updateActionInstruction(state,'step_right',1200,{...evidence,fresh:false,resolved:true,sideConfirmed:true});
  assert.equal(state.guidance,'step_left');assert.equal(state.resolvedReadings,0);
  state=updateActionInstruction(state,'aligned',2400,{...evidence,fresh:false,passingFrame:true});
  assert.equal(state.guidance,'steady');
});

test('capture keeps its own shared hold and only passing frames permit the accepted prompt',()=>{
  let state=confirmed(),hold=emptyCaptureHold();
  for(const now of [1200,1600,2000]) {
    hold=updateCaptureHold(hold,now,true);
    state=updateActionInstruction(state,hold.confirmed?'aligned':'steady',now,{...evidence,resolved:true,passingFrame:true});
    assert.equal(hold.ready,now===2000);
    assert.equal(state.guidance,now===1200?'step_left':'aligned');
  }
});

test('resolution checks the instructed axis in raw and filtered measurements, including classifier conflicts',()=>{
  const m:AlignmentMeasurements={centerX:.5,area:.08,rollDegrees:0,widthDifference:.12,heightDifference:0,perspectiveDifference:.12};
  const resolved=(g:Parameters<typeof correctionResolved>[0],a=m,b=m,side=false,high=false)=>correctionResolved(g,a,b,DEFAULT_ALIGNMENT_REFERENCE,side,high);
  assert.equal(resolved('left'),true);assert.equal(resolved('left',{...m,centerX:.39}),false);
  assert.equal(resolved('step_left',m,m,true),false);assert.equal(resolved('step_left',m,{...m,heightDifference:-.09}),false);
  assert.equal(resolved('lower_phone'),true);assert.equal(resolved('lower_phone',m,{...m,widthDifference:.18}),false);
  assert.equal(resolved('lower_phone',m,m,false,true),false);
  assert.equal(resolved('farther'),true);assert.equal(resolved('farther',{...m,area:.23}),false);
});
