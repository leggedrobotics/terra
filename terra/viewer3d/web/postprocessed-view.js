import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { metricCell, postprocessedRoutes } from './postprocessed.js';

export const REGION_COLORS = { dig: 0xe69f00, finish: 0x009e73, dump: 0x8b5cf6, deposit: 0xc084fc };
const ROUTE_COLORS = { checked: 0x2563eb, failed: 0xdc2626, missing: 0xd97706, unverified: 0xd97706 };

/** Refined masks and returned paths, in the metric renderer's map frame. */
export class PostprocessedOverlay {
  constructor(view, data) {
    this.view = view; this.scene = view.scene; this.data = data;
    this.regions = new THREE.Group(); this.routeGroup = new THREE.Group();
    view.overlay.add(this.regions, this.routeGroup);
    this.workspaces = new Map(data.workspaces.map(w => [w.id, w]));
    this.routes = postprocessedRoutes(data);
  }

  point(x, y, lift = .04) {
    const [row, col] = metricCell(x, y, this.data.grid);
    return this.scene.point(row, col).setY(this.scene.heightAt(row, col) + lift);
  }

  clear(group) {
    group.traverse(child => { child.geometry?.dispose(); if (child.material) { this.view.lineMaterials.delete(child.material); child.material.dispose(); } });
    group.clear();
  }

  update(index) {
    const event = this.view.episode.event(index), workspace = this.workspaces.get(event.workspace_id);
    const key = `${event.workspace_id}:${index}`;
    this.regions.visible = this.view.workspacesVisible;
    this.routeGroup.visible = this.view.routesVisible;
    const { native, loose } = this.scene.frame.metric;
    const sameTerrain = this.native?.every((row, i) => row === native[i]) && this.loose?.every((row, i) => row === loose[i]);
    const routeKeys = [...(workspace ? this.routes.byWorkspace.get(workspace.id) ?? [] : this.routes.byFrame.get(index) ?? [])];
    const routeKey = JSON.stringify(routeKeys);
    if (sameTerrain && (key === this.key || (!event.reservations?.length && !this.hadReservations && event.workspace_id === this.workspaceId && routeKey === this.routeKey))) { this.key = key; return; }
    this.key = key; this.workspaceId = event.workspace_id;
    this.routeKey = routeKey;
    this.native = native; this.loose = loose;
    this.hadReservations = !!event.reservations?.length;
    this.clear(this.regions); this.clear(this.routeGroup);
    const reservations = event.reservations ?? (workspace?.reservation_geometry ? [{ agent_id: workspace.agent_id, geometry: workspace.reservation_geometry }] : []);
    for (const reservation of reservations) this.geometry(reservation.geometry, event.conflicts?.length ? 0xdc2626 : [0x2563eb, 0x009e73, 0xa855f7, 0xd97706][reservation.agent_id % 4], this.regions);
    if (workspace?.refined_geometry) this.geometry(workspace.refined_geometry, 0xe69f00, this.regions);
    for (const key of routeKeys) for (const segment of this.routes.routes.get(key).segments) {
      if (segment.points.length < 2) continue;
      const material = new LineMaterial({ color: ROUTE_COLORS[segment.status], linewidth: 3.2, transparent: true, opacity: .95, dashed: segment.status !== 'checked' || segment.relocate, dashSize: .45, gapSize: .25, depthTest: false, depthWrite: false });
      this.view.lineMaterials.add(material);
      const geometry = new LineGeometry(); geometry.setPositions(segment.points.flatMap(([x, y]) => this.point(x, y).toArray()));
      const line = new Line2(geometry, material); line.computeLineDistances(); line.renderOrder = 28; line.frustumCulled = false;
      this.routeGroup.add(line);
    }
    if (!workspace) { this.view.resize(); return; }
    const { rows, cols, resolution_m: tile } = this.data.grid;
    for (const [name, ink] of Object.entries(REGION_COLORS)) {
      const runs = workspace.masks?.[name] ?? [], cells = new Set();
      for (const [row, col, width] of runs) for (let c = col; c < col + width; c++) cells.add(row * cols + c);
      const surface = [], edges = [], half = tile / 2;
      const member = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && cells.has(r * cols + c);
      for (const i of cells) {
        const row = Math.floor(i / cols), col = i % cols, p = this.scene.point(row, col), h = this.scene.heightAt(row, col) + .012 + Object.keys(REGION_COLORS).indexOf(name) * .003;
        const a = [p.x - half, h, p.z - half], b = [p.x + half, h, p.z - half], c = [p.x + half, h, p.z + half], d = [p.x - half, h, p.z + half];
        surface.push(...a, ...d, ...b, ...b, ...d, ...c);
        // Increasing row points toward scene -Z in metric mode.
        if (!member(row + 1, col)) edges.push(...a, ...b);
        if (!member(row - 1, col)) edges.push(...d, ...c);
        if (!member(row, col - 1)) edges.push(...a, ...d);
        if (!member(row, col + 1)) edges.push(...b, ...c);
      }
      if (!surface.length) continue;
      const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(surface, 3));
      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: ink, transparent: true, opacity: name === 'deposit' ? .09 : .2, side: THREE.DoubleSide, depthWrite: false }));
      mesh.renderOrder = 25; mesh.userData.skipAO = true; this.regions.add(mesh);
      const outline = new THREE.BufferGeometry(); outline.setAttribute('position', new THREE.Float32BufferAttribute(edges, 3));
      const line = new THREE.LineSegments(outline, new THREE.LineBasicMaterial({ color: ink, transparent: true, opacity: .95, depthTest: false, depthWrite: false }));
      line.renderOrder = 26; this.regions.add(line);
    }
    this.view.resize();
  }

  geometry(geometry, color, group) {
    if (!geometry) return;
    if (geometry.type === 'Feature') return this.geometry(geometry.geometry, color, group);
    if (geometry.type === 'GeometryCollection') { for (const item of geometry.geometries) this.geometry(item, color, group); return; }
    if (geometry.type === 'MultiPolygon') { for (const coordinates of geometry.coordinates) this.geometry({ type: 'Polygon', coordinates }, color, group); return; }
    const rings = geometry.type === 'Polygon' ? geometry.coordinates : geometry.type === 'LineString' ? [geometry.coordinates] : [];
    for (const ring of rings) {
      if (ring.length < 2) continue;
      const material = new LineMaterial({ color, linewidth: 2.8, transparent: true, opacity: .95, depthTest: false, depthWrite: false });
      this.view.lineMaterials.add(material);
      const shape = new LineGeometry(); shape.setPositions(ring.flatMap(([x, y]) => this.point(x, y, .055).toArray()));
      const line = new Line2(shape, material); line.renderOrder = 29; line.frustumCulled = false; group.add(line);
    }
  }

  bounds() {
    const box = new THREE.Box3(), data = this.data;
    const add = (x, y, radius = 1) => { const p = this.point(x, y); box.expandByPoint(p.clone().add(new THREE.Vector3(-radius, -.5, -radius))); box.expandByPoint(p.clone().add(new THREE.Vector3(radius, 2, radius))); };
    const yaw = data.grid.yaw_rad ?? 0, c = Math.cos(yaw), s = Math.sin(yaw), world = (row, col) => [data.grid.origin_xy_m[0] + (col * c - row * s) * data.grid.resolution_m, data.grid.origin_xy_m[1] + (col * s + row * c) * data.grid.resolution_m];
    for (const [row, col, width] of data.grid.target ?? []) {
      add(...world(row, col));
      add(...world(row, col + width - 1));
    }
    for (const workspace of data.workspaces) if (workspace.pose) add(workspace.pose[0], workspace.pose[1], 3);
    for (const agent of data.initial.agents) add(agent.pose[0], agent.pose[1], 3);
    return box;
  }

  dispose() { this.clear(this.regions); this.clear(this.routeGroup); this.regions.removeFromParent(); this.routeGroup.removeFromParent(); }
}
