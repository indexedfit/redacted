// Service Worker: pre-cache local app shell, local ESM, and local model
// WASM files are loaded dynamically based on SIMD support
const VERSION = 'v4-dynamic-wasm';
const CACHE = `face-app-${VERSION}`;

const CORE_ASSETS = [
  '/',
  '/index.html',
  '/app.js',
  '/manifest.webmanifest',
  // Local library
  '/lib/tasks-vision.js',
  // Local model
  '/models/face_detector.tflite'
];

// WASM files that will be cached on-demand
const WASM_ASSETS = [
  '/wasm/vision_wasm_internal.js',
  '/wasm/vision_wasm_internal.wasm',
  '/wasm/vision_wasm_nosimd_internal.js',
  '/wasm/vision_wasm_nosimd_internal.wasm'
];

self.addEventListener('install', (e) => {
  // Only pre-cache core assets, WASM files loaded on-demand
  e.waitUntil(caches.open(CACHE).then(cache => cache.addAll(CORE_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method === 'GET') {
    e.respondWith(
      caches.match(req).then(hit => {
        if (hit) return hit;
        
        // Fetch and cache the resource
        return fetch(req).then(resp => {
          // Only cache successful responses
          if (resp.status === 200) {
            const copy = resp.clone();
            caches.open(CACHE).then(cache => {
              cache.put(req, copy).catch(() => {});
            });
          }
          return resp;
        }).catch(() => {
          // Offline fallback
          if (req.mode === 'navigate') return caches.match('/index.html');
          return Promise.reject('offline and not cached');
        });
      })
    );
  }
});
