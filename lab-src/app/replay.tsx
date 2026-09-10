"use client";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
export type SceneData = {
  kind?: string;
  follow_body?: number;
  bodies: string[];
  meshes: {
    body: number;
    vertices: number[];
    indices: number[];
    color: number[];
  }[];
  table: { size: number[]; position: number[] } | null;
  cube_size: number;
};
export type Sample = {
  t: number;
  phase: string;
  cube: number[];
  tips: number[][];
  temperature: number;
  torque: number;
  requested_torque: number;
  current: number;
  saturation: number;
  grip: number[][];
  contact_forces?: number[][];
  failures?: number;
};
export type Recording = { poses: Float32Array; samples: Sample[]; result: any };
const sceneCache = new Map<string, Promise<SceneData>>();
export function loadScene(url: string) {
  if (!sceneCache.has(url))
    sceneCache.set(
      url,
      fetch(url).then((r) => {
        if (!r.ok) throw Error("Scene unavailable");
        return r.json();
      }),
    );
  return sceneCache.get(url)!;
}
export function Replay({
  sceneData,
  recording,
  playback,
  cameraView,
  showPath,
  label,
}: {
  sceneData: SceneData;
  recording: Recording;
  playback: { current: { time: number } };
  cameraView: string;
  showPath: boolean;
  label: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<any>(null);
  const current = useRef({ playback, recording, showPath, cameraView });
  current.current = { playback, recording, showPath, cameraView };
  const [error, setError] = useState("");
  useEffect(() => {
    if (!host.current) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      setError(
        "3D rendering is unavailable on this device. Use the video view to watch this recording.",
      );
      return;
    }
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#e6e9e1");
    scene.fog = new THREE.Fog("#e6e9e1", 5, 13);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    host.current.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(36, 1, 0.02, 30);
    camera.up.set(0, 0, 1);
    camera.position.set(1.95, 1.8, 1.55);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.37, 0, 0.5);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.25;
    controls.maxDistance = 5;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.update();
    scene.add(new THREE.HemisphereLight("#f6f9ed", "#738171", 2.2));
    const light = new THREE.DirectionalLight("#fff6e5", 4.2);
    light.position.set(1, -2, 4);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    light.shadow.camera.left = -2;
    light.shadow.camera.right = 2;
    light.shadow.camera.top = 2;
    light.shadow.camera.bottom = -2;
    light.shadow.normalBias = 0.002;
    scene.add(light);
    const fill = new THREE.DirectionalLight("#d6e7ff", 1.8);
    fill.position.set(-2, 1, 2);
    scene.add(fill);
    const bodies = sceneData.bodies.map(() => {
      const g = new THREE.Group();
      scene.add(g);
      return g;
    });
    const geometries: THREE.BufferGeometry[] = [];
    const materials: THREE.Material[] = [];
    for (const m of sceneData.meshes) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(m.vertices, 3));
      geo.setIndex(m.indices);
      geo.computeVertexNormals();
      geometries.push(geo);
      const finger = m.body % 11 >= 9;
      const joint = m.body % 11 === 7;
      const material = new THREE.MeshStandardMaterial({
        color:
          sceneData.kind === "locomotion"
            ? new THREE.Color(...(m.color as [number, number, number]))
            : finger
              ? "#343b36"
              : joint
                ? "#5d6860"
                : m.body < 11
                  ? "#ced4c4"
                  : "#f3f0e4",
        metalness: finger ? 0.25 : 0.3,
        roughness: 0.36,
      });
      materials.push(material);
      const mesh = new THREE.Mesh(geo, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      bodies[m.body].add(mesh);
    }
    if (sceneData.cube_size > 0) {
      const cubeGeo = new THREE.BoxGeometry(
        sceneData.cube_size,
        sceneData.cube_size,
        sceneData.cube_size,
      );
      geometries.push(cubeGeo);
      const cubeMats = ["#ea712d", "#c84020", "#f8cc61", "#e9a137", "#faf0d1", "#478972"].map(
        (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.48 }),
      );
      materials.push(...cubeMats);
      const cube = new THREE.Mesh(cubeGeo, cubeMats);
      cube.castShadow = true;
      cube.receiveShadow = true;
      bodies[bodies.length - 1].add(cube);
      const edgeGeo = new THREE.EdgesGeometry(cubeGeo);
      geometries.push(edgeGeo);
      const edgeMat = new THREE.LineBasicMaterial({
        color: "#5c341c",
        transparent: true,
        opacity: 0.5,
      });
      materials.push(edgeMat);
      cube.add(new THREE.LineSegments(edgeGeo, edgeMat));
    }
    function box(size: number[], pos: number[], color: string) {
      const g = new THREE.BoxGeometry(...(size as [number, number, number]));
      geometries.push(g);
      const m = new THREE.MeshStandardMaterial({ color, roughness: 0.75 });
      materials.push(m);
      const b = new THREE.Mesh(g, m);
      b.position.set(...(pos as [number, number, number]));
      b.receiveShadow = true;
      scene.add(b);
      return b;
    }
    if (sceneData.table) box(sceneData.table.size, sceneData.table.position, "#b7c1b1");
    box([30, 30, 0.03], [0, 0, -0.03], "#e6e9e1");
    const grid = new THREE.GridHelper(40, 200, "#a8b7a4", "#cbd3c5");
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.002;
    scene.add(grid);
    if (sceneData.kind !== "locomotion")
      for (const [x, y, c] of [
        [0.45, -0.3, "#e1c17e"],
        [0.48, 0.3, "#438973"],
      ] as [number, number, string][]) {
        const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.12, 0.12, 0.001));
        geometries.push(g);
        const m = new THREE.LineBasicMaterial({ color: c });
        materials.push(m);
        const l = new THREE.LineSegments(g, m);
        l.position.set(x, y, 0.301);
        scene.add(l);
      }
    const traceGeo = new THREE.BufferGeometry();
    geometries.push(traceGeo);
    const traceMat = new THREE.LineBasicMaterial({
      color: "#c37033",
      transparent: true,
      opacity: 0.6,
    });
    materials.push(traceMat);
    const trace =
      sceneData.kind === "locomotion"
        ? new THREE.LineSegments(traceGeo, traceMat)
        : new THREE.Line(traceGeo, traceMat);
    scene.add(trace);
    const glows = (sceneData.kind === "locomotion" ? [] : [9, 10, 20, 21]).map((i) => {
      const g = new THREE.SphereGeometry(0.011, 12, 8);
      geometries.push(g);
      const m = new THREE.MeshBasicMaterial({
        color: "#64efaa",
        transparent: true,
        opacity: 0.7,
        depthTest: false,
      });
      materials.push(m);
      const dot = new THREE.Mesh(g, m);
      dot.visible = false;
      bodies[i].add(dot);
      return dot;
    });
    let lastRecording: Recording | null = null;
    let raf = 0;
    function animate() {
      const c = current.current;
      const count = sceneData.bodies.length;
      const max = c.recording.result.frames - 1;
      const f = Math.max(
        0,
        Math.min(max, c.playback.current.time * (c.recording.result.fps || 50)),
      );
      const f0 = Math.floor(f),
        f1 = Math.min(max, f0 + 1),
        mix = c.recording.samples[f0]?.failures !== c.recording.samples[f1]?.failures ? 0 : f - f0;
      const data = c.recording.poses;
      for (let b = 0; b < count; b++) {
        const a = (f0 * count + b) * 7,
          z = (f1 * count + b) * 7;
        const g = bodies[b];
        g.position.set(
          THREE.MathUtils.lerp(data[a], data[z], mix),
          THREE.MathUtils.lerp(data[a + 1], data[z + 1], mix),
          THREE.MathUtils.lerp(data[a + 2], data[z + 2], mix),
        );
        g.quaternion.set(data[a + 4], data[a + 5], data[a + 6], data[a + 3]);
        g.quaternion.slerp(
          new THREE.Quaternion(data[z + 4], data[z + 5], data[z + 6], data[z + 3]),
          mix,
        );
      }
      if (lastRecording !== c.recording) {
        trace.geometry.dispose();
        trace.geometry = new THREE.BufferGeometry().setFromPoints(
          sceneData.kind === "locomotion"
            ? c.recording.samples.slice(1).flatMap((s, i) => {
                const prev = c.recording.samples[i];
                const a = prev.failures === s.failures ? prev : s;
                return [a, s].map(
                  (p) => new THREE.Vector3(...(p.cube.slice(0, 3) as [number, number, number])),
                );
              })
            : c.recording.samples.map(
                (s) => new THREE.Vector3(...(s.cube.slice(0, 3) as [number, number, number])),
              ),
        );
        lastRecording = c.recording;
      }
      trace.visible = c.showPath;
      trace.geometry.setDrawRange(
        0,
        sceneData.kind === "locomotion" ? f0 * 2 : Math.max(2, f0 + 1),
      );
      const contacts = c.recording.samples[f0]?.contact_forces;
      glows.forEach((g, i) => {
        const v = contacts?.[i] || [];
        g.visible = Math.hypot(...v) > 0.15;
      });
      if (c.cameraView === "follow") {
        const target = bodies[sceneData.follow_body ?? bodies.length - 1].position;
        camera.position.add(new THREE.Vector3().subVectors(target, controls.target));
        controls.target.copy(target);
      }
      if (sceneData.kind === "locomotion") {
        const pos = bodies[0].position;
        light.position.set(pos.x + 1, pos.y - 2, 4);
        light.target.position.copy(pos);
        light.target.updateMatrixWorld();
      }
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }
    const observer = new ResizeObserver(() => {
      if (!host.current) return;
      const w = host.current.clientWidth,
        h = host.current.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.zoom = Math.min(1, camera.aspect / (sceneData.kind === "locomotion" ? 0.9 : 1.5));
      camera.updateProjectionMatrix();
    });
    observer.observe(host.current);
    state.current = { camera, controls };
    animate();
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      geometries.forEach((x) => x.dispose());
      trace.geometry.dispose();
      materials.forEach((x) => x.dispose());
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      renderer.domElement.remove();
      state.current = null;
    };
  }, [sceneData]);
  useEffect(() => {
    const s = state.current;
    if (!s) return;
    const views: Record<string, number[][]> = {
      overview: [
        [1.95, 1.8, 1.55],
        [0.37, 0, 0.5],
      ],
      handoff: [
        [1.0, 0.6, 0.94],
        [0.48, 0, 0.64],
      ],
      follow: [
        [1.0, 0.6, 0.94],
        [0.48, 0, 0.64],
      ],
      front: [
        [2.3, 0, 1.14],
        [0.35, 0, 0.48],
      ],
      top: [
        [0.4, 0.001, 2.8],
        [0.4, 0, 0.3],
      ],
    };
    const v =
      sceneData.kind === "locomotion"
        ? cameraView === "top"
          ? [
              [0, 0.01, 4],
              [0, 0, 0.5],
            ]
          : cameraView === "front"
            ? [
                [3, 0, 1.2],
                [0, 0, 0.65],
              ]
            : [
                [2.4, 2.4, 1.8],
                [0, 0, 0.65],
              ]
        : views[cameraView] || views.overview;
    s.camera.position.set(...v[0]);
    s.controls.target.set(...v[1]);
    s.controls.update();
  }, [cameraView, sceneData]);
  return (
    <div
      className="replay-canvas"
      ref={host}
      role="img"
      aria-label={`${label}. Interactive recorded robot trajectory. Drag to orbit, scroll to zoom.`}
    >
      {error && (
        <p className="render-error" role="alert">
          {error}
        </p>
      )}
      <div className="canvas-label">{label}</div>
    </div>
  );
}
