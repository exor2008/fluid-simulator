import * as THREE from "three";

import { GUI } from "three/addons/libs/lil-gui.module.min.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VolumeRenderShader1 } from "three/addons/shaders/VolumeShader.js";
import WebGL from "three/addons/capabilities/WebGL.js";

if (WebGL.isWebGL2Available() === false) {
  document.body.appendChild(WebGL.getWebGL2ErrorMessage());
}

let renderer,
  scene,
  camera,
  controls,
  material,
  volconfig,
  cmtextures,
  voxelData,
  texture,
  sizeX,
  sizeY,
  sizeZ;

function init() {
  scene = new THREE.Scene();

  volconfig = {
    clim1: 0,
    clim2: 1,
    renderstyle: "mip",
    isothreshold: 0.15,
    colormap: "viridis",
  };

  initRenderer();
  initCamera();
  initCameraControl();
  initGui();
  initVoxelData();
  initTexture();
  initMaterial();
  initMesh();

  render();
  window.addEventListener("resize", onWindowResize);
}

function initRenderer() {
  // Create renderer
  renderer = new THREE.WebGLRenderer();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);
}

function initCamera() {
  const h = 512; // frustum height
  const aspect = window.innerWidth / window.innerHeight;
  camera = new THREE.OrthographicCamera(
    (-h * aspect) / 2,
    (h * aspect) / 2,
    h / 2,
    -h / 2,
    1,
    5000
  );

  // camera.zoom = 3;
  camera.up.set(0, 0, 1); // In our data, z is up
  camera.position.set(0, 0, 100);

  // camera.top -= 65;
  // camera.bottom -= 65;

  return camera;
}

function initCameraControl() {
  // Create controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", render);
  controls.target.set(0, 0, 0);
  controls.minZoom = 0.5;
  controls.maxZoom = 6;
  controls.enablePan = true;
  controls.update();
}

function initGui() {
  const gui = new GUI();
  gui.add(volconfig, "clim1", 0, 1, 0.01).onChange(updateUniforms);
  gui.add(volconfig, "clim2", 0, 1, 0.01).onChange(updateUniforms);
  gui
    .add(volconfig, "colormap", { gray: "gray", viridis: "viridis" })
    .onChange(updateUniforms);
  gui
    .add(volconfig, "renderstyle", { mip: "mip", iso: "iso" })
    .onChange(updateUniforms);
  gui.add(volconfig, "isothreshold", 0, 1, 0.01).onChange(updateUniforms);
}
function initVoxelData() {
  voxelData = new Float32Array(sizeX * sizeY * sizeZ);
}

function initTexture() {
  texture = new THREE.Data3DTexture(voxelData, sizeX, sizeY, sizeZ);
  texture.format = THREE.RedFormat;
  texture.type = THREE.FloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
}

function initMaterial() {
  cmtextures = {
    viridis: new THREE.TextureLoader().load("cm_viridis.png", render),
    gray: new THREE.TextureLoader().load("cm_gray.png", render),
  };

  // Material
  const shader = VolumeRenderShader1;

  const uniforms = THREE.UniformsUtils.clone(shader.uniforms);

  uniforms["u_data"].value = texture;
  uniforms["u_size"].value.set(sizeX, sizeY, sizeZ);
  uniforms["u_clim"].value.set(volconfig.clim1, volconfig.clim2);
  uniforms["u_renderstyle"].value = volconfig.renderstyle == "mip" ? 0 : 1; // 0: MIP, 1: ISO
  uniforms["u_renderthreshold"].value = volconfig.isothreshold; // For ISO renderstyle
  uniforms["u_cmdata"].value = cmtextures[volconfig.colormap];

  material = new THREE.ShaderMaterial({
    uniforms: uniforms,
    vertexShader: shader.vertexShader,
    fragmentShader: shader.fragmentShader,
    side: THREE.BackSide, // The volume shader uses the backface as its "reference point"
  });
}

function initMesh() {
  const geometry = new THREE.BoxGeometry(sizeX, sizeY, sizeZ);
  geometry.translate(sizeX / 2 - 0.5, sizeY / 2 - 0.5, sizeZ / 2 - 0.5);

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
}

function updateUniforms() {
  material.uniforms["u_clim"].value.set(volconfig.clim1, volconfig.clim2);
  material.uniforms["u_renderstyle"].value =
    volconfig.renderstyle == "mip" ? 0 : 1; // 0: MIP, 1: ISO
  material.uniforms["u_renderthreshold"].value = volconfig.isothreshold; // For ISO renderstyle
  material.uniforms["u_cmdata"].value = cmtextures[volconfig.colormap];
  render();
}

function onWindowResize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  const aspect = window.innerWidth / window.innerHeight;
  const frustumHeight = camera.top - camera.bottom;

  camera.left = (-frustumHeight * aspect) / 2;
  camera.right = (frustumHeight * aspect) / 2;

  camera.updateProjectionMatrix();

  render();
}

function render() {
  renderer.render(scene, camera);
}

function animate() {
  texture.image.data.set(voxelData);
  texture.needsUpdate = true;

  renderer.render(scene, camera);
}

async function streamVoxelData() {
  // TODO: deal with uncontrolled getReader
  try {
    const response = await fetch("http://127.0.0.1:8000/fluid/stream", {
      headers: {
        "Content-Type": "application/octet-stream",
      },
    });

    const reader = response.body.getReader();

    while (true) {
      let totalBytesRead = 0;
      let chunkSize = sizeX * sizeY * sizeZ * Float32Array.BYTES_PER_ELEMENT;
      let chunk = new Uint8Array(chunkSize);

      while (totalBytesRead < chunkSize) {
        let { done, value } = await reader.read();

        if (done) {
          return;
        }

        if (value.byteLength + totalBytesRead < chunkSize) {
          chunk.set(value, totalBytesRead);
          totalBytesRead += value.byteLength;
          continue;
        }

        if (value.byteLength + totalBytesRead == chunkSize) {
          chunk.set(value, totalBytesRead);
          voxelData = new Float32Array(chunk.buffer);
          requestAnimationFrame(animate);
          break;
        }

        while (value.byteLength + totalBytesRead > chunkSize) {
          let toSend;
          if (totalBytesRead > 0) {
            toSend = value.subarray(0, chunkSize - totalBytesRead);
            value = value.subarray(
              chunkSize - totalBytesRead,
              value.byteLength
            );
            chunk.set(toSend, totalBytesRead);
            totalBytesRead = 0;
          } else {
            toSend = value.subarray(0, chunkSize);
            value = value.subarray(chunkSize, value.byteLength);
            chunk.set(toSend, 0);
          }
          voxelData = new Float32Array(chunk.buffer);
          requestAnimationFrame(animate);
        }
        totalBytesRead = value.byteLength;
      }
    }
  } catch (error) {
    console.error("Error fetching data:", error);
  }
}

async function fetchSize() {
  fetch("http://127.0.0.1:8000/size")
    .then((response) => response.json())
    .then((size) => {
      sizeX = size.x;
      sizeY = size.y;
      sizeZ = size.z;

      init();
      streamVoxelData();
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

fetchSize();
