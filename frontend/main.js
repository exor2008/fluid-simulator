import * as THREE from "three";

import { GUI } from "three/addons/libs/lil-gui.module.min.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VolumeRenderShader1 } from "three/addons/shaders/VolumeShader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
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
  sizeZ,
  gltfObj;

function init() {
  scene = new THREE.Scene();

  volconfig = {
    clim1: 0,
    clim2: 1,
    renderstyle: "mip",
    isothreshold: 0.15,
    colormap: "viridis",
    pause: onPauseClick,
    resume: onResumeClick,
    reset: onResetClick,
    data: "smoke",
    gltf_visible: true,
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

  renderer.getContext().enable(renderer.getContext().DEPTH_TEST);
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
    2000
  );

  // camera.zoom = 3;
  camera.up.set(0, 0, 1); // In our data, z is up
  camera.position.set(300, 0, 50);

  // camera.top -= 65;
  // camera.bottom -= 65;

  return camera;
}

function initCameraControl() {
  // Create controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", render);
  controls.target.set(100, 100, 50);
  controls.minZoom = 0.5;
  controls.maxZoom = 6;
  controls.enablePan = true;
  controls.update();
}

function initGui() {
  const gui = new GUI();
  gui.add(volconfig, "clim1", 0.0, 10.0, 0.01).onChange(updateUniforms);
  gui.add(volconfig, "clim2", 0.0, 10.0, 0.01).onChange(updateUniforms);
  gui
    .add(volconfig, "colormap", [
      "viridis",
      "hot",
      "gray",
      "blues",
      "cividis",
      "cool",
      "copper",
      "inferno",
      "plasma",
      "seismic",
      "winter",
      "ylgn",
    ])
    .onChange(updateUniforms);
  gui
    .add(volconfig, "renderstyle", { mip: "mip", iso: "iso" })
    .onChange(updateUniforms);
  gui.add(volconfig, "isothreshold", 0, 1, 0.01).onChange(updateUniforms);
  gui.add(volconfig, "pause");
  gui.add(volconfig, "resume");
  gui.add(volconfig, "reset");
  gui
    .add(volconfig, "data", [
      "smoke",
      "pressure",
      "block",
      "speed",
      "speed_smoke",
      "smoke_block",
    ])
    .onChange(swithcData);
  gui.add(volconfig, "gltf_visible", true).onChange(gltfVisibleSwitch);
}

function onPauseClick() {
  fetch("http://127.0.0.1:8000/pause", { method: "POST" });
}

function onResumeClick() {
  fetch("http://127.0.0.1:8000/resume", { method: "POST" });
}

function onResetClick() {
  fetch("http://127.0.0.1:8000/reset", { method: "POST" });
}

function swithcData() {
  const url = `http://127.0.0.1:8000/switch/${volconfig.data}`;
  fetch(url, { method: "POST" });
}

function updateGravity() {
  const url = `http://127.0.0.1:8000/gravity/${volconfig.gravity}`;
  fetch(url, { method: "POST" });
}

function gltfVisibleSwitch() {
  gltfObj.scene.visible = volconfig.gltf_visible;
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
  texture.depthTest = true;
}

function initMaterial() {
  cmtextures = {
    viridis: new THREE.TextureLoader().load("cm_viridis.png", render),
    hot: new THREE.TextureLoader().load("cm_hot.png", render),
    gray: new THREE.TextureLoader().load("cm_gray.png", render),
    blues: new THREE.TextureLoader().load("cm_blues.png", render),
    cividis: new THREE.TextureLoader().load("cm_cividis.png", render),
    cool: new THREE.TextureLoader().load("cm_cool.png", render),
    copper: new THREE.TextureLoader().load("cm_copper.png", render),
    inferno: new THREE.TextureLoader().load("cm_inferno.png", render),
    plasma: new THREE.TextureLoader().load("cm_plasma.png", render),
    seismic: new THREE.TextureLoader().load("cm_seismic.png", render),
    winter: new THREE.TextureLoader().load("cm_winter.png", render),
    ylgn: new THREE.TextureLoader().load("cm_ylgn.png", render),
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
    depthWrite: true,
    depthTest: true,
    side: THREE.BackSide, // The volume shader uses the backface as its "reference point"
  });
}

function initMesh() {
  const geometry = new THREE.BoxGeometry(sizeX, sizeY, sizeZ);
  geometry.translate(sizeX / 2 - 0.5, sizeY / 2 - 0.5, sizeZ / 2 - 0.5);

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
}

function initGltf(file) {
  // Read and show gltf model
  const reader = new FileReader();
  reader.onload = function (event) {
    const result = event.target.result;
    loadModel(result);
  };
  reader.readAsDataURL(file);
}

function loadModel(dataURL) {
  const loader = new GLTFLoader();
  loader.load(
    dataURL,
    async function (gltf) {
      const ambientLight = new THREE.AmbientLight(0xffffff); // Ambient light
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1); // Directional light
      directionalLight.position.set(300, 0, 50).normalize();
      directionalLight.lookAt(100, 100, 50);
      scene.add(directionalLight);

      gltf.scene.scale.set(10, 10, 10);

      await renderer.compileAsync(gltf.scene, camera, scene);
      scene.add(gltf.scene);
      gltfObj = gltf;
    },
    undefined,
    function (error) {
      console.error("Error loading model:", error);
    }
  );
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
        console.log("while 1"); // workaround. block execution otherwise
        setTimeout(function () {}, 0); // workaround. block execution otherwise
        let { done, value } = await reader.read();

        if (done) {
          return;
        }

        if (value.byteLength + totalBytesRead < chunkSize) {
          chunk.set(value, totalBytesRead);
          totalBytesRead += value.byteLength;
        } else if (value.byteLength + totalBytesRead == chunkSize) {
          chunk.set(value, totalBytesRead);
          voxelData = new Float32Array(chunk.buffer);
          requestAnimationFrame(animate);
          break;
        } else {
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
    }
  } catch (error) {
    console.error("Error fetching data:", error);
  }
}

function showModal() {
  document.getElementById("modal").style.display = "block";
}

function hideModal() {
  document.getElementById("modal").style.display = "none";
}

function submitForm() {
  const x = document.getElementById("x").value;
  const y = document.getElementById("y").value;
  const z = document.getElementById("z").value;
  const fileInput = document.getElementById("file");

  // Assume you have a function to upload the file using XMLHttpRequest
  uploadInitForm(fileInput.files[0], { x, y, z });

  // Close the modal after submission
  hideModal();
}

function uploadInitForm(file, data) {
  const formData = new FormData();
  formData.append("gltf", file);
  formData.append("x", data.x);
  formData.append("y", data.y);
  formData.append("z", data.z);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "http://127.0.0.1:8000/init", true);

  // Set up event listeners for completion or error
  xhr.onload = function () {
    if (xhr.status === 200) {
      sizeX = data.x;
      sizeY = data.y;
      sizeZ = data.z;

      init();
      initGltf(file);

      streamVoxelData();
    } else {
      console.error("File upload failed");
    }
  };

  xhr.onerror = function () {
    console.error("Network error during file upload");
  };

  // Send the FormData object
  xhr.send(formData);
}

document.getElementById("submitBtn").addEventListener("click", submitForm);
showModal();
// fetchSize();
