import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { makeMachine } from './models.js';

const agent = { id: 0, type: 0, action_type: 0, width: 7, height: 7, reach: [4, 11], loaded: 27, wheel_angle: 0, shovel_lifted: 0, cabin_yaw: .45 };
const closePoint = (actual, expected, description) => assert.ok(actual.distanceTo(expected) < 1e-8, `${description}: ${actual.toArray()} != ${expected.toArray()}`);
const worldPosition = object => object.getWorldPosition(new THREE.Vector3());
const worldAxis = (object, axis) => axis.clone().applyQuaternion(object.getWorldQuaternion(new THREE.Quaternion()));

test('excavator bucket reverses by yaw beneath a separate curl hinge and carries upright', () => {
  const machine = makeMachine(agent, .5, { labels: false });
  try {
    machine.setPose(agent, true);
    const orientation = machine.root.getObjectByName('bucket-orientation');
    const curl = machine.root.getObjectByName('bucket-curl');
    assert.equal(orientation.parent, curl);
    assert.equal(orientation.rotation.y, Math.PI); assert.equal(orientation.rotation.x, 0); assert.equal(orientation.rotation.z, 0);
    assert.equal(curl.rotation.y, 0); assert.equal(curl.rotation.x, 0);
    const forward = new THREE.Vector3(Math.cos(agent.cabin_yaw), 0, -Math.sin(agent.cabin_yaw));
    assert.ok(worldAxis(orientation, new THREE.Vector3(1, 0, 0)).dot(forward) < -.99, 'the cutting edge must face inward');
    assert.ok(worldAxis(orientation, new THREE.Vector3(0, 1, 0)).y > .99, 'carried soil must remain upright');
    assert.ok(machine.root.getObjectByName('bucket-soil').position.y < 0, 'the load sits inside the bowl below its top hinge');
  } finally { machine.dispose(); }
});

test('hinge, fixed-length bucket links and hydraulic pins remain connected across all work poses', () => {
  const machine = makeMachine(agent, .5, { labels: false });
  machine.root.position.set(9, 2, -4); machine.root.rotation.y = -.7;
  try {
    const hinge = machine.root.getObjectByName('bucket-hinge'), curl = machine.root.getObjectByName('bucket-curl'), stick = machine.root.getObjectByName('stick-pivot');
    for (const kind of ['', 'dig', 'dump']) for (let index = 0; index <= 40; index++) {
      machine.setPose({ ...agent, cabin_yaw: agent.cabin_yaw + index * .01, loaded: kind === 'dump' ? 0 : 27 }, true, index / 40, kind);
      machine.root.updateWorldMatrix(true, true);
      closePoint(worldPosition(curl), worldPosition(hinge), `${kind} bucket hinge`);
      const rig = machine.bucketRig;
      const destination = stick.worldToLocal(worldPosition(rig.destination));
      assert.ok(Math.abs(rig.origin.position.distanceTo(rig.joint.position) - rig.firstLength) < 1e-8, 'rocker length must not stretch');
      assert.ok(Math.abs(rig.joint.position.distanceTo(destination) - rig.secondLength) < 1e-8, 'bucket link length must not stretch');
      for (const link of rig.links) {
        const start = rig.origin.position.clone(), middle = rig.joint.position.clone(), end = destination.clone();
        start.z = middle.z = end.z = link.z;
        closePoint(link.first.localToWorld(new THREE.Vector3(0, -.5, 0)), stick.localToWorld(start), 'rocker start pin');
        closePoint(link.first.localToWorld(new THREE.Vector3(0, .5, 0)), stick.localToWorld(middle.clone()), 'rocker middle pin');
        closePoint(link.second.localToWorld(new THREE.Vector3(0, -.5, 0)), stick.localToWorld(middle), 'link middle pin');
        closePoint(link.second.localToWorld(new THREE.Vector3(0, .5, 0)), stick.localToWorld(end), 'link bucket pin');
      }
      for (const actuator of machine.hydraulics) {
        closePoint(actuator.barrel.localToWorld(new THREE.Vector3(0, -.5, 0)), worldPosition(actuator.start), 'hydraulic barrel pin');
        closePoint(actuator.piston.localToWorld(new THREE.Vector3(0, .5, 0)), worldPosition(actuator.end), 'hydraulic piston pin');
      }
      assert.ok(worldAxis(machine.root.getObjectByName('bucket-orientation'), new THREE.Vector3(0, 1, 0)).y > .55, 'bucket must not turn upside down during curl');
    }
  } finally { machine.dispose(); }
});

test('skid-steer bucket keeps its forward orientation', () => {
  const loader = { ...agent, type: 2, cabin_yaw: 0 };
  const machine = makeMachine(loader, .5, { labels: false });
  try {
    machine.setPose(loader, true);
    const bucket = machine.root.getObjectByName('loader-bucket');
    assert.equal(bucket.rotation.y, 0);
    assert.equal(machine.root.getObjectByName('bucket-orientation'), undefined);
    assert.ok(worldAxis(bucket, new THREE.Vector3(1, 0, 0)).x > .95);
  } finally { machine.dispose(); }
});

test('excavator bucket stays compact across machine footprints and map scales', () => {
  for (const [width, height, tile] of [[7, 7, .5], [5, 9, .6875], [9, 5, 1.25]]) {
    const state = { ...agent, width, height }, machine = makeMachine(state, tile, { labels: false });
    try {
      const bucket = machine.root.getObjectByName('bucket-orientation').clone(true);
      // Measure the complete asset, including teeth, ears, pins and payload,
      // without the arm pose or cabin rotation inflating its bounding box.
      bucket.rotation.set(0, 0, 0);
      const size = new THREE.Box3().setFromObject(bucket, true).getSize(new THREE.Vector3());
      const S = Math.min(width, height) * tile, W = width * tile;
      assert.ok(size.x > S * .48 && size.x < S * .54, 'bucket length must not dominate the machine');
      assert.ok(size.y > S * .38 && size.y < S * .43, 'bowl and mounting ears stay near cab height');
      assert.ok(size.z > W * .24 && size.z < W * .28, 'bucket stays about one quarter of track width');
      const load = machine.root.getObjectByName('bucket-soil');
      assert.ok(load.position.y + load.scale.y < 0, 'scaled load remains below the hinge');
      const positiveEar = machine.root.getObjectByName('bucket-ear-1'), negativeEar = machine.root.getObjectByName('bucket-ear--1');
      positiveEar.geometry.computeBoundingBox();
      const earBounds = positiveEar.geometry.boundingBox;
      const gap = positiveEar.position.z + earBounds.min.z - negativeEar.position.z - earBounds.max.z;
      const outerSpan = positiveEar.position.z + earBounds.max.z - negativeEar.position.z - earBounds.min.z;
      assert.ok(gap > machine.root.getObjectByName('bucket-hinge-housing').scale.y + S * .01, 'ear bevels clear the unchanged stick-eye housing');
      assert.ok(machine.root.getObjectByName('bucket-main-pin').scale.y > outerSpan, 'hinge pin passes through both mounting ears');
      machine.setPose(state, true, .5, 'dig');
      closePoint(worldPosition(machine.root.getObjectByName('bucket-curl')), worldPosition(machine.root.getObjectByName('bucket-hinge')), 'scaled bucket hinge');
    } finally { machine.dispose(); }
  }
});

test('rounded excavator panels preserve their specified footprint dimensions', () => {
  const machine = makeMachine(agent, .5, { labels: false });
  try {
    const body = machine.root.getObjectByName('excavator-upper-body');
    assert.equal(body.geometry.type, 'RoundedBoxGeometry'); assert.ok(body.geometry.parameters.radius > 0);
    body.geometry.computeBoundingBox(); const size = body.geometry.boundingBox.getSize(new THREE.Vector3());
    assert.ok(Math.abs(size.x - agent.height * .5 * .65) < 1e-6);
    assert.ok(Math.abs(size.z - agent.width * .5 * .66) < 1e-6);
  } finally { machine.dispose(); }
});
