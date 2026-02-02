/**
 * behaviors.js - Transformer Explorer Demo
 *
 * Vanilla ES6 JavaScript. No frameworks.
 * Dependencies (loaded before this file): Plotly.js, D3.js v7, gpt-tokenizer.
 *
 * Data files expected in ./demo_data/:
 *   tokenizer_examples.json, embeddings_projected.json,
 *   embeddings_full.json, attention_data.json
 */

document.addEventListener('DOMContentLoaded', async () => {
  'use strict';

  // ============================================================
  // 1. CONSTANTS & CONFIGURATION
  // ============================================================

  const DATA_BASE = './demo_data';
  const DATA_FILES = {
    tokenizer: `${DATA_BASE}/tokenizer_examples.json`,
    projected: `${DATA_BASE}/embeddings_projected.json`,
    full:      `${DATA_BASE}/embeddings_full.json`,
    attention: `${DATA_BASE}/attention_data.json`,
  };

  const NEIGHBOR_COUNT       = 10;
  const ANALOGY_RESULT_COUNT = 5;
  const HEATMAP_COLOR_SCALE  = d3.interpolateBlues;
  const QKV_COLORS           = { q: '#E74C3C', k: '#2980B9', v: '#27AE60' };
  const GENERAL_DOT_COLOR    = '#333333';
  const SELECTED_DOT_COLOR   = '#E74C3C';
  const SELECTED_DOT_SIZE    = 16;
  const DEFAULT_DOT_SIZE     = 4;
  const CATEGORY_DOT_SIZE    = 7;
  const ANIMATION_STEP_MS    = 600;
  const EMBED_3D_CAMERA_DEFAULT = {
    eye: { x: 1.5, y: 1.5, z: 1.5 },
    center: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 0, z: 1 },
  };
  // Only prevent true eye==center degeneration; don't interfere with intentional close zoom.
  const EMBED_3D_CAMERA_MIN_DISTANCE = 1e-6;

  // ============================================================
  // 2. APPLICATION STATE
  // ============================================================

  const state = {
    data: { tokenizer: null, projected: null, full: null, attention: null },

    tokenizer: {
      currentCategory: null,
      currentExampleIndex: null,
    },

    onehot: {
      currentStep: 0,
      totalSteps: 0,
      displayedWords: [],
    },

    embeddings: {
      wordList: [],
      norms: {},
      selectedWord: null,
      activeCaseFiles: new Set(),
      plotInitialized: false,
      plotHandlersBound: false,
      mouseGuardsBound: false,
      cameraCorrectionInFlight: false,
      lastCameraCorrectionSig: null,
      lastKnownGoodCamera: null,
      pendingSelectionTimer: null,
      currentDim: '2d',
      baseColors: [],
      baseSizes: [],
      baseTexts: [],
    },

    attention: {
      currentSentenceId: null,
      currentLayer: 0,
      currentHead: 0,
      selectedTokenIndex: null,
      temperature: 1.0,
      qkvVisible: false,
      projectionLayer: 0,
      animationTimer: null,
      heatmapRendered: false,
      projectionInitialized: false,
    },

    viewInitialized: {
      tokenizer: false,
      onehot: false,
      embeddings: false,
      attention: false,
    },
  };

  // ============================================================
  // 3. DOM REFERENCES
  // ============================================================

  const $ = (id) => document.getElementById(id);

  const dom = {
    // Loader
    appLoader:       $('app-loader'),
    loaderStatus:    $('loader-status'),
    loaderProgress:  $('loader-progress'),
    app:             $('app'),
    sessionBadge:    $('app-session-badge'),

    // Tabs
    viewTabs:        $('view-tabs'),

    // Tokenizer
    tokInput:        $('tokenizer-input'),
    tokSubmit:       $('tokenizer-submit'),
    tokClear:        $('tokenizer-clear'),
    tokCategories:   $('tokenizer-categories'),
    tokExamplesList: $('tokenizer-examples-list'),
    tokExamplesTbody:$('tokenizer-examples-tbody'),
    tokOutput:       $('tokenizer-output'),
    tokStats:        $('tokenizer-stats'),
    tokCompToggle:   $('tokenizer-comparison-toggle'),
    tokComp:         $('tokenizer-comparison'),
    tokCompGpt2:     $('tokenizer-comparison-gpt2-tokens'),
    tokCompBert:     $('tokenizer-comparison-bert-tokens'),
    tokAnnotationArea: $('tokenizer-annotation-area'),
    tokAnnotation:   $('tokenizer-annotation'),

    // One-hot
    ohPrev:          $('onehot-prev'),
    ohNext:          $('onehot-next'),
    ohStepIndicator: $('onehot-step-indicator'),
    ohVis:           $('onehot-visualization'),
    ohNarration:     $('onehot-narration'),

    // Embeddings
    embSearch:       $('embedding-search'),
    embSearchBtn:    $('embedding-search-btn'),
    embResetBtn:     $('embedding-reset-btn'),
    embOovNotice:    $('embedding-oov-notice'),
    embCaseFiles:    $('embedding-casefiles'),
    embDim2d:        $('embedding-dim-2d'),
    embDim3d:        $('embedding-dim-3d'),
    embPlot:         $('embedding-plot'),
    embNeighborsList:$('embedding-neighbors-list'),
    embSelectedWord: $('embedding-selected-word'),
    embAnalogyA:     $('embedding-analogy-a'),
    embAnalogyB:     $('embedding-analogy-b'),
    embAnalogyC:     $('embedding-analogy-c'),
    embAnalogySubmit:$('embedding-analogy-submit'),
    embAnalogyResult:$('embedding-analogy-result'),
    embAnalogyNbrs:  $('embedding-analogy-neighbors'),
    embAnalogyNbrsList: $('embedding-analogy-neighbors-list'),

    // Attention
    attSentenceSelect: $('attention-sentence-select'),
    attLayerSelect:    $('attention-layer-select'),
    attHeadSelect:     $('attention-head-select'),
    attBestBtn:        $('attention-best-btn'),
    attTokens:         $('attention-tokens'),
    attSelectedToken:  $('attention-selected-token'),
    attStaticNbrs:     $('attention-static-neighbors'),
    attHeatmap:        $('attention-heatmap'),
    attContextualNbrs: $('attention-contextual-neighbors'),
    attQkvToggle:      $('attention-qkv-toggle'),
    attQkvDisplay:     $('attention-qkv-display'),
    attQkvQ:           $('attention-qkv-query'),
    attQkvK:           $('attention-qkv-key'),
    attQkvV:           $('attention-qkv-value'),
    attTemp:           $('attention-temperature'),
    attTempValue:      $('attention-temperature-value'),
    attPlayBtn:        $('attention-play-btn'),
    attLayerSlider:    $('attention-layer-slider'),
    attLayerSliderVal: $('attention-layer-slider-value'),
    attProjection:     $('attention-projection'),

    // Shared
    tooltip:           $('tooltip'),
    tooltipText:       $('tooltip-text'),
  };


  // ============================================================
  // 4. UTILITY FUNCTIONS
  // ============================================================

  // ---- Math ----

  function dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
  }

  function vectorNorm(v) {
    let sum = 0;
    for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
    return Math.sqrt(sum);
  }

  function cosineSimilarity(a, b, normA, normB) {
    if (normA === undefined) normA = vectorNorm(a);
    if (normB === undefined) normB = vectorNorm(b);
    if (normA < 1e-10 || normB < 1e-10) return 0;
    return dotProduct(a, b) / (normA * normB);
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  function vectorAdd(a, b) {
    return a.map((v, i) => v + b[i]);
  }

  function vectorSub(a, b) {
    return a.map((v, i) => v - b[i]);
  }

  function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  function cloneCamera(camera) {
    if (!camera) return null;
    return {
      eye: {
        x: camera.eye?.x,
        y: camera.eye?.y,
        z: camera.eye?.z,
      },
      center: {
        x: camera.center?.x,
        y: camera.center?.y,
        z: camera.center?.z,
      },
      up: {
        x: camera.up?.x,
        y: camera.up?.y,
        z: camera.up?.z,
      },
    };
  }

  function sanitize3dCamera(camera, fallbackCamera = EMBED_3D_CAMERA_DEFAULT) {
    const fallback = fallbackCamera || EMBED_3D_CAMERA_DEFAULT;
    const eye = camera?.eye || {};
    const center = camera?.center || {};
    const up = camera?.up || {};

    // Ignore transient/partial relayout payloads to avoid snapping the view.
    if (
      !isFiniteNumber(eye.x) || !isFiniteNumber(eye.y) || !isFiniteNumber(eye.z) ||
      !isFiniteNumber(center.x) || !isFiniteNumber(center.y) || !isFiniteNumber(center.z)
    ) {
      return { camera: null, changed: false, valid: false };
    }

    const safeCamera = {
      eye: { x: eye.x, y: eye.y, z: eye.z },
      center: { x: center.x, y: center.y, z: center.z },
      up: {
        x: isFiniteNumber(up.x) ? up.x : fallback.up.x,
        y: isFiniteNumber(up.y) ? up.y : fallback.up.y,
        z: isFiniteNumber(up.z) ? up.z : fallback.up.z,
      },
    };

    const dx = safeCamera.eye.x - safeCamera.center.x;
    const dy = safeCamera.eye.y - safeCamera.center.y;
    const dz = safeCamera.eye.z - safeCamera.center.z;
    const dist = Math.hypot(dx, dy, dz);
    if (!Number.isFinite(dist)) {
      return { camera: null, changed: false, valid: false };
    }

    let changed = false;
    if (dist < EMBED_3D_CAMERA_MIN_DISTANCE) {
      changed = true;
      if (dist < 1e-8) {
        const dirX = fallback.eye.x - fallback.center.x;
        const dirY = fallback.eye.y - fallback.center.y;
        const dirZ = fallback.eye.z - fallback.center.z;
        const dirNorm = Math.hypot(dirX, dirY, dirZ) || 1;
        const scale = EMBED_3D_CAMERA_MIN_DISTANCE / dirNorm;
        safeCamera.eye = {
          x: safeCamera.center.x + dirX * scale,
          y: safeCamera.center.y + dirY * scale,
          z: safeCamera.center.z + dirZ * scale,
        };
      } else {
        const scale = EMBED_3D_CAMERA_MIN_DISTANCE / dist;
        safeCamera.eye = {
          x: safeCamera.center.x + dx * scale,
          y: safeCamera.center.y + dy * scale,
          z: safeCamera.center.z + dz * scale,
        };
      }
    }

    return { camera: safeCamera, changed, valid: true };
  }

  function cameraSignature(camera) {
    if (!camera || !camera.eye || !camera.center) return 'invalid';
    const up = camera.up || EMBED_3D_CAMERA_DEFAULT.up;
    return [
      camera.eye.x, camera.eye.y, camera.eye.z,
      camera.center.x, camera.center.y, camera.center.z,
      up.x, up.y, up.z,
    ].map(v => Number(v).toFixed(6)).join(',');
  }

  function swallowPlotlyRejection(maybePromise, context) {
    if (maybePromise && typeof maybePromise.catch === 'function') {
      maybePromise.catch((err) => {
        if (err !== undefined) {
          console.warn(`${context}:`, err);
        }
      });
    }
    return maybePromise;
  }

  // ---- Nearest neighbors (for embeddings_full) ----

  function findNeighbors(queryVec, topN = NEIGHBOR_COUNT, exclude = new Set()) {
    const qNorm = vectorNorm(queryVec);
    const { wordList, norms } = state.embeddings;
    const vectors = state.data.full.vectors;
    const results = [];

    for (let i = 0; i < wordList.length; i++) {
      const w = wordList[i];
      if (exclude.has(w)) continue;
      const sim = cosineSimilarity(queryVec, vectors[w], qNorm, norms[w]);
      results.push({ word: w, similarity: sim });
    }

    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topN);
  }

  // ---- Color helpers ----

  function tokenColor(tokenId) {
    const hue = (tokenId * 137.508) % 360;
    return `hsl(${hue}, 55%, 82%)`;
  }

  function tokenColorDark(tokenId) {
    const hue = (tokenId * 137.508) % 360;
    return `hsl(${hue}, 50%, 35%)`;
  }

  // ---- DOM helpers ----

  function show(el) { el.classList.remove('is-hidden'); }
  function hide(el) { el.classList.add('is-hidden'); }
  function toggle(el) { el.classList.toggle('is-hidden'); }

  function clearChildren(el) { el.innerHTML = ''; }

  function createTokenChip(text, tokenId, extraClasses = '') {
    const span = document.createElement('span');
    span.className = `tag is-medium ${extraClasses}`.trim();
    span.dataset.tokenId = tokenId;
    span.style.backgroundColor = tokenColor(tokenId);
    span.style.color = tokenColorDark(tokenId);
    span.style.margin = '2px';
    span.style.cursor = 'default';

    // Display text - make whitespace visible
    let display = text;
    if (text === ' ') display = '␣';
    else if (text === '\n') display = '↵';
    else if (text === '\t') display = '⇥';
    span.textContent = display;
    span.title = `ID: ${tokenId}`;
    return span;
  }

  function renderNeighborList(container, neighbors) {
    clearChildren(container);
    if (!neighbors || neighbors.length === 0) {
      container.innerHTML = '<li class="has-text-grey-light is-italic">No data</li>';
      return;
    }
    neighbors.forEach(nb => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${nb.word}</span> <span class="has-text-grey is-size-7">${nb.similarity.toFixed(3)}</span>`;
      container.appendChild(li);
    });
  }

  function showTooltip(x, y, html) {
    dom.tooltipText.innerHTML = html;
    dom.tooltip.style.left = (x + 12) + 'px';
    dom.tooltip.style.top = (y - 10) + 'px';
    show(dom.tooltip);
  }

  function hideTooltip() {
    hide(dom.tooltip);
  }


  // ============================================================
  // 5. DATA LOADING
  // ============================================================

  async function loadData() {
    const entries = Object.entries(DATA_FILES);
    let loaded = 0;

    dom.loaderProgress.value = 0;
    dom.loaderProgress.max = entries.length;

    const promises = entries.map(async ([key, url]) => {
      try {
        dom.loaderStatus.textContent = `Loading ${key}…`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
        state.data[key] = await resp.json();
        loaded++;
        dom.loaderProgress.value = loaded;
        dom.loaderStatus.textContent = `Loaded ${loaded}/${entries.length}…`;
      } catch (err) {
        console.error(`Failed to load ${url}:`, err);
        dom.loaderStatus.textContent = `Error loading ${key}. Check console.`;
        throw err;
      }
    });

    await Promise.all(promises);
    dom.loaderStatus.textContent = 'Initializing…';
  }


  // ============================================================
  // 6. TAB / VIEW SWITCHING
  // ============================================================

  function switchView(viewName) {
    // Update tabs
    dom.viewTabs.querySelectorAll('li').forEach(li => {
      li.classList.toggle('is-active', li.dataset.view === viewName);
    });

    // Update panels
    document.querySelectorAll('.view-panel').forEach(panel => {
      const id = panel.id.replace('view-', '');
      panel.classList.toggle('is-hidden', id !== viewName);
    });

    // Update session badge
    const activeTab = dom.viewTabs.querySelector(`li[data-view="${viewName}"]`);
    if (activeTab) {
      dom.sessionBadge.textContent = `Session ${activeTab.dataset.session}`;
    }

    // Lazy initialization
    if (!state.viewInitialized[viewName]) {
      switch (viewName) {
        case 'tokenizer':  initTokenizer();  break;
        case 'onehot':     initOneHot();     break;
        case 'embeddings': initEmbeddings(); break;
        case 'attention':  initAttention();  break;
      }
      state.viewInitialized[viewName] = true;
    }

    // Plotly needs a resize after becoming visible
    if (viewName === 'embeddings' && state.embeddings.plotInitialized) {
      Plotly.Plots.resize(dom.embPlot);
    }
    if (viewName === 'attention' && state.attention.projectionInitialized) {
      Plotly.Plots.resize(dom.attProjection);
    }
  }

  function bindTabEvents() {
    dom.viewTabs.querySelectorAll('li').forEach(li => {
      li.addEventListener('click', () => switchView(li.dataset.view));
    });
  }


  // ============================================================
  // 7. TOKENIZER VIEW
  // ============================================================

  // ---- Live tokenizer detection ----
  const liveTokenizer = (() => {
    const candidates = [
      'cl100k_base',
      'GPTTokenizer_cl100k_base',
      'GPTTokenizer',
      'gptTokenizer',
    ];
    for (const name of candidates) {
      const obj = window[name];
      if (!obj) continue;
      if (typeof obj.encode === 'function') {
        console.log(`Live tokenizer found: window.${name}`);
        return obj;
      }
      if (obj.default && typeof obj.default.encode === 'function') {
        console.log(`Live tokenizer found: window.${name}.default`);
        return obj.default;
      }
    }
    console.warn('Live tokenizer not detected - using pre-computed examples only.');
    return null;
  })();

  function tokenizeLive(text) {
    if (!liveTokenizer) return null;
    try {
      const ids = liveTokenizer.encode(text);
      return ids.map(id => {
        let decoded;
        try { decoded = liveTokenizer.decode([id]); }
        catch { decoded = `[${id}]`; }
        return { id, text: decoded };
      });
    } catch (err) {
      console.error('Live tokenization failed:', err);
      return null;
    }
  }

  function getPrimaryTokensForText(text, fallbackTokens = null) {
    const liveResult = tokenizeLive(text);
    if (liveResult) return liveResult;
    return fallbackTokens;
  }

  function initTokenizer() {
    const data = state.data.tokenizer;
    const expectedVocabSize = data?.metadata?.primaryTokenizer?.vocabSize;
    const liveVocabSize = liveTokenizer?.vocabularySize;
    if (
      Number.isFinite(expectedVocabSize) &&
      Number.isFinite(liveVocabSize) &&
      expectedVocabSize !== liveVocabSize
    ) {
      console.warn(
        `Tokenizer vocab mismatch: expected ${expectedVocabSize} from demo data, ` +
        `but live tokenizer reports ${liveVocabSize}.`
      );
    }

    // Build category buttons
    const categories = data.categories;
    const sortedCats = Object.entries(categories)
      .sort((a, b) => a[1].order - b[1].order);

    clearChildren(dom.tokCategories);
    sortedCats.forEach(([catId, meta]) => {
      const btn = document.createElement('button');
      btn.className = 'button';
      btn.dataset.category = catId;
      btn.textContent = `${meta.icon} ${meta.label}`;
      btn.addEventListener('click', () => loadTokenizerCategory(catId));
      dom.tokCategories.appendChild(btn);
    });

    // Event listeners
    dom.tokSubmit.addEventListener('click', handleTokenize);
    dom.tokInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleTokenize();
      }
    });
    dom.tokClear.addEventListener('click', () => {
      dom.tokInput.value = '';
      clearTokenizerOutput();
    });
    dom.tokCompToggle.addEventListener('click', () => {
      toggle(dom.tokComp);
      dom.tokCompToggle.textContent =
        dom.tokComp.classList.contains('is-hidden') ? 'Show Comparison' : 'Hide Comparison';
    });

    // Load the opener example automatically
    const openerIdx = data.examples.findIndex(e => e.isOpener);
    if (openerIdx >= 0) {
      const opener = data.examples[openerIdx];
      dom.tokInput.value = opener.text;
      const openerTokens = getPrimaryTokensForText(opener.text, opener.primary.tokens);
      renderTokenizerResult(openerTokens, opener.text);
      renderTokenizerComparison(opener);
      showAnnotation(opener.annotation);
    }
  }

  function handleTokenize() {
    const text = dom.tokInput.value;
    if (!text) { clearTokenizerOutput(); return; }

    // Try live tokenizer first
    const liveResult = tokenizeLive(text);
    if (liveResult) {
      renderTokenizerResult(liveResult, text);
      // Check if this matches a curated example for annotation + comparison
      const match = state.data.tokenizer.examples.find(e => e.text === text);
      if (match) {
        renderTokenizerComparison(match);
        showAnnotation(match.annotation);
      } else {
        hideAnnotation();
        clearChildren(dom.tokCompGpt2);
        clearChildren(dom.tokCompBert);
      }
      return;
    }

    // Fallback: check curated examples
    const match = state.data.tokenizer.examples.find(e => e.text === text);
    if (match) {
      renderTokenizerResult(match.primary.tokens, text);
      renderTokenizerComparison(match);
      showAnnotation(match.annotation);
    } else {
      dom.tokOutput.innerHTML = '<p class="has-text-warning">Live tokenizer unavailable and no pre-computed match. Try a curated example.</p>';
      dom.tokStats.textContent = '';
    }
  }

  function renderTokenizerResult(tokens, originalText) {
    clearChildren(dom.tokOutput);
    tokens.forEach(tok => {
      dom.tokOutput.appendChild(createTokenChip(tok.text, tok.id));
    });

    const charCount = originalText.length;
    const ratio = charCount > 0 ? (charCount / tokens.length).toFixed(1) : '-';
    dom.tokStats.textContent = `${tokens.length} tokens · ${charCount} chars · ${ratio} chars/token`;
  }

  function renderTokenizerComparison(example) {
    if (!example.comparison) return;

    // GPT-2
    clearChildren(dom.tokCompGpt2);
    const gpt2 = example.comparison.gpt2;
    if (gpt2 && gpt2.tokens) {
      gpt2.tokens.forEach(tok => {
        dom.tokCompGpt2.appendChild(createTokenChip(tok.text, tok.id));
      });
    }

    // BERT
    clearChildren(dom.tokCompBert);
    const bert = example.comparison.bert;
    if (bert && bert.tokens) {
      bert.tokens.forEach(tok => {
        dom.tokCompBert.appendChild(createTokenChip(tok.text, tok.id));
      });
    }
  }

  function loadTokenizerCategory(catId) {
    state.tokenizer.currentCategory = catId;

    // Highlight active button
    dom.tokCategories.querySelectorAll('button').forEach(btn => {
      btn.classList.toggle('is-primary', btn.dataset.category === catId);
    });

    // Populate table
    const examples = state.data.tokenizer.examples
      .map((ex, idx) => ({ ...ex, _idx: idx }))
      .filter(ex => ex.category === catId);

    clearChildren(dom.tokExamplesTbody);
    examples.forEach(ex => {
      const tr = document.createElement('tr');
      tr.dataset.exampleIndex = ex._idx;
      tr.style.cursor = 'pointer';

      const tdText = document.createElement('td');
      const displayText = ex.text.length > 60 ? ex.text.slice(0, 57) + '…' : ex.text;
      tdText.textContent = displayText || '(empty)';

      const tdCount = document.createElement('td');
      tdCount.textContent = ex.primary.tokenCount;

      tr.appendChild(tdText);
      tr.appendChild(tdCount);
      tr.addEventListener('click', () => loadTokenizerExample(ex._idx));
      dom.tokExamplesTbody.appendChild(tr);
    });

    show(dom.tokExamplesList);
  }

  function loadTokenizerExample(idx) {
    const example = state.data.tokenizer.examples[idx];
    state.tokenizer.currentExampleIndex = idx;

    dom.tokInput.value = example.text;
    const exampleTokens = getPrimaryTokensForText(example.text, example.primary.tokens);
    renderTokenizerResult(exampleTokens, example.text);
    renderTokenizerComparison(example);
    showAnnotation(example.annotation);

    // Highlight active row
    dom.tokExamplesTbody.querySelectorAll('tr').forEach(tr => {
      tr.classList.toggle('is-selected', parseInt(tr.dataset.exampleIndex) === idx);
    });
  }

  function clearTokenizerOutput() {
    dom.tokOutput.innerHTML = '<p class="has-text-grey-light is-italic">Enter text above and click Tokenize</p>';
    dom.tokStats.textContent = '';
    hideAnnotation();
    clearChildren(dom.tokCompGpt2);
    clearChildren(dom.tokCompBert);
  }

  function showAnnotation(text) {
    dom.tokAnnotation.textContent = text;
    show(dom.tokAnnotationArea);
  }

  function hideAnnotation() {
    hide(dom.tokAnnotationArea);
  }


  // ============================================================
  // 8. ONE-HOT ENCODING VIEW
  // ============================================================

  function initOneHot() {
    const sequence = state.data.tokenizer.oneHotDemo.animationSequence;
    state.onehot.totalSteps = sequence.length;
    state.onehot.currentStep = 0;

    dom.ohPrev.addEventListener('click', () => stepOneHot(-1));
    dom.ohNext.addEventListener('click', () => stepOneHot(1));

    renderOneHotStep(0);
  }

  function stepOneHot(delta) {
    const newStep = state.onehot.currentStep + delta;
    if (newStep < 0 || newStep >= state.onehot.totalSteps) return;
    state.onehot.currentStep = newStep;
    renderOneHotStep(newStep);
  }

  function renderOneHotStep(stepIndex) {
    const oneHot = state.data.tokenizer.oneHotDemo;
    const sequence = oneHot.animationSequence;
    const step = sequence[stepIndex];

    // Update controls
    dom.ohStepIndicator.textContent = `Step ${stepIndex + 1} of ${state.onehot.totalSteps}`;
    dom.ohPrev.disabled = stepIndex === 0;
    dom.ohNext.disabled = stepIndex === state.onehot.totalSteps - 1;

    // Update narration
    dom.ohNarration.textContent = step.narration;

    // Track which words to display
    if (step.action === 'show_vector') {
      if (!state.onehot.displayedWords.includes(step.word)) {
        state.onehot.displayedWords.push(step.word);
      }
    }

    // Render visualization
    const container = dom.ohVis;
    clearChildren(container);

    const svgWidth = container.clientWidth - 40 || 700;
    const rowHeight = 80;
    const vocabSize = oneHot.vocabSize;

    if (step.action === 'show_vector' || step.action === 'compute_dot_product') {
      const wordsToShow = step.action === 'compute_dot_product'
        ? [step.wordA, step.wordB]
        : [...state.onehot.displayedWords];

      const svgHeight = wordsToShow.length * rowHeight + (step.action === 'compute_dot_product' ? 80 : 20);

      const svg = d3.select(container).append('svg')
        .attr('width', svgWidth)
        .attr('height', svgHeight);

      wordsToShow.forEach((word, rowIdx) => {
        renderOneHotVector(svg, word, oneHot.words[word], rowIdx, svgWidth, rowHeight, vocabSize);
      });

      // Dot product result
      if (step.action === 'compute_dot_product') {
        const dp = oneHot.dotProducts.find(d => d.wordA === step.wordA && d.wordB === step.wordB);
        const hasDotProduct = dp && Number.isFinite(dp.dotProduct);
        const y = wordsToShow.length * rowHeight + 10;

        svg.append('line')
          .attr('x1', 80).attr('x2', svgWidth - 40)
          .attr('y1', y).attr('y2', y)
          .attr('stroke', '#999').attr('stroke-width', 1);

        svg.append('text')
          .attr('x', svgWidth / 2).attr('y', y + 30)
          .attr('text-anchor', 'middle')
          .attr('font-size', '16px')
          .attr('font-weight', 'bold')
          .attr('fill', hasDotProduct ? (dp.dotProduct === 0 ? '#E74C3C' : '#27AE60') : '#777')
          .text(hasDotProduct ? `Dot product = ${dp.dotProduct}` : 'Dot product unavailable');

        if (hasDotProduct && dp.dotProduct === 0) {
          svg.append('text')
            .attr('x', svgWidth / 2).attr('y', y + 52)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('fill', '#888')
            .text('The "1" positions don\'t overlap → every product is 0');
        } else if (!hasDotProduct) {
          svg.append('text')
            .attr('x', svgWidth / 2).attr('y', y + 52)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('fill', '#888')
            .text('Demo data is missing this pair in oneHot.dotProducts.');
        }
      }

    } else if (step.action === 'reveal_problem') {
      const div = document.createElement('div');
      div.className = 'has-text-centered p-5';
      div.innerHTML = `
        <p class="is-size-4 has-text-danger has-text-weight-bold mb-4">The Problem</p>
        <p class="is-size-5 mb-3">Every pair of different words has similarity <strong>0</strong>.</p>
        <p class="is-size-6 mb-3">
          "king" \u2219 "queen" = 0 &nbsp;&nbsp;←&nbsp; same as &nbsp;→&nbsp; "king" \u2219 "refrigerator" = 0
        </p>
        <p class="is-size-6 has-text-grey">
          No structure. No notion of closeness. No meaning.<br>
          We need something better: <strong>embeddings</strong>.
        </p>
      `;
      container.appendChild(div);
    }
  }

  function renderOneHotVector(svg, word, wordData, rowIdx, svgWidth, rowHeight, vocabSize) {
    const y = rowIdx * rowHeight + 15;
    const barLeft = 120;
    const barRight = svgWidth - 60;
    const barWidth = barRight - barLeft;
    const barHeight = 28;
    const tokenId = wordData.primaryTokenId;

    // Word label
    svg.append('text')
      .attr('x', 10).attr('y', y + barHeight / 2 + 5)
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text(`"${word}"`);

    // Full bar (zeros background)
    svg.append('rect')
      .attr('x', barLeft).attr('y', y)
      .attr('width', barWidth).attr('height', barHeight)
      .attr('fill', '#F0F0F0').attr('stroke', '#CCC').attr('rx', 3);

    // Position of the "1" (proportional within bar)
    const oneX = barLeft + (tokenId / vocabSize) * barWidth;
    const markerW = Math.max(barWidth / 60, 4);

    // Highlight the "1" slot
    svg.append('rect')
      .attr('x', oneX - markerW / 2).attr('y', y)
      .attr('width', markerW).attr('height', barHeight)
      .attr('fill', '#E74C3C');

    // "0 0 0 ... 1 ... 0 0 0" label inside bar
    svg.append('text')
      .attr('x', barLeft + 8).attr('y', y + barHeight / 2 + 4)
      .attr('font-size', '10px').attr('fill', '#AAA')
      .text('0  0  0  0 ···');

    svg.append('text')
      .attr('x', barRight - 8).attr('y', y + barHeight / 2 + 4)
      .attr('font-size', '10px').attr('fill', '#AAA')
      .attr('text-anchor', 'end')
      .text('··· 0  0  0  0');

    // Marker label below
    svg.append('text')
      .attr('x', oneX).attr('y', y + barHeight + 14)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px').attr('fill', '#E74C3C')
      .text(`↑ 1 at position ${tokenId}`);

    // Dimension label
    svg.append('text')
      .attr('x', barRight + 5).attr('y', y + barHeight / 2 + 4)
      .attr('font-size', '9px').attr('fill', '#AAA')
      .text(`${vocabSize.toLocaleString()}d`);
  }


  // ============================================================
  // 9. EMBEDDING SPACE VIEW
  // ============================================================

  function initEmbeddings() {
    const projData = state.data.projected;
    const fullData = state.data.full;

    // Pre-compute word list and norms for neighbor search
    state.embeddings.wordList = Object.keys(fullData.vectors);
    state.embeddings.norms = {};
    for (const w of state.embeddings.wordList) {
      state.embeddings.norms[w] = vectorNorm(fullData.vectors[w]);
    }

    // Check if 3D data is available
    const has3d = projData.metadata.has3d === true ||
                  (projData.words.length > 0 && projData.words[0].z3d !== undefined);
    if (!has3d) {
      dom.embDim3d.disabled = true;
      dom.embDim3d.parentElement.title = '3D projection not available in data';
      dom.embDim3d.parentElement.style.opacity = '0.4';
    }

    // Build case-file buttons
    clearChildren(dom.embCaseFiles);
    Object.entries(projData.categories).forEach(([catId, meta]) => {
      if (!meta.isCaseFile) return;
      const btn = document.createElement('button');
      btn.className = 'button';
      btn.dataset.casefile = catId;
      btn.textContent = meta.label;
      btn.style.borderColor = meta.color;
      btn.style.borderWidth = '2px';
      btn.addEventListener('click', () => toggleCaseFile(catId));
      dom.embCaseFiles.appendChild(btn);
    });

    // Build the scatter plot
    renderEmbeddingPlot();
    bindEmbeddingMouseGuards();

    // Event listeners
    dom.embSearchBtn.addEventListener('click', () => embeddingSearch());
    dom.embSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') embeddingSearch();
    });
    dom.embResetBtn.addEventListener('click', embeddingReset);
    dom.embAnalogySubmit.addEventListener('click', computeAnalogy);
    dom.embAnalogyA.addEventListener('keydown', (e) => { if (e.key === 'Enter') computeAnalogy(); });
    dom.embAnalogyB.addEventListener('keydown', (e) => { if (e.key === 'Enter') computeAnalogy(); });
    dom.embAnalogyC.addEventListener('keydown', (e) => { if (e.key === 'Enter') computeAnalogy(); });

    // 2D/3D toggle
    dom.embDim2d.addEventListener('change', () => {
      state.embeddings.currentDim = '2d';
      renderEmbeddingPlot();
    });
    dom.embDim3d.addEventListener('change', () => {
      state.embeddings.currentDim = '3d';
      renderEmbeddingPlot();
    });
  }

  function renderEmbeddingPlot() {
    const projData = state.data.projected;
    const words = projData.words;
    const is3d = state.embeddings.currentDim === '3d';

    // Build arrays for Plotly
    const xs = [], ys = [], zs = [], texts = [], colors = [], sizes = [];

    words.forEach(w => {
      if (is3d) {
        xs.push(w.x3d);
        ys.push(w.y3d);
        zs.push(w.z3d);
      } else {
        xs.push(w.x);
        ys.push(w.y);
      }
      texts.push(w.word);
      colors.push(GENERAL_DOT_COLOR);
      sizes.push(DEFAULT_DOT_SIZE);
    });

    state.embeddings.baseColors = [...colors];
    state.embeddings.baseSizes  = [...sizes];
    state.embeddings.baseTexts  = [...texts];

    let trace, layout;

    if (is3d) {
      trace = {
        x: xs, y: ys, z: zs,
        mode: 'markers',
        type: 'scatter3d',
        text: texts,
        hoverinfo: 'text',
        marker: {
          color: colors,
          size: sizes.map(s => s * 0.8),
          opacity: 0.6,
          line: { width: 0 },
        },
      };
      layout = {
        hovermode: 'closest',
        scene: {
          xaxis: { visible: false },
          yaxis: { visible: false },
          zaxis: { visible: false },
          aspectmode: 'data',
          camera: { ...EMBED_3D_CAMERA_DEFAULT },
        },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
      };
    } else {
      trace = {
        x: xs, y: ys,
        mode: 'markers',
        type: 'scatter',
        text: texts,
        hoverinfo: 'text',
        marker: {
          color: colors,
          size: sizes,
          opacity: 0.6,
          line: { width: 0 },
        },
      };
      layout = {
        hovermode: 'closest',
        dragmode: 'pan',
        xaxis: { visible: false },
        yaxis: { visible: false, scaleanchor: 'x' },
        margin: { l: 10, r: 10, t: 10, b: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      };
    }

    const config = {
      responsive: true,
      scrollZoom: true,
      displayModeBar: false,
    };

    if (!state.embeddings.plotInitialized) {
      Plotly.newPlot(dom.embPlot, [trace], layout, config);
      state.embeddings.plotInitialized = true;
    } else {
      Plotly.react(dom.embPlot, [trace], layout, config);
    }

    if (is3d) {
      state.embeddings.lastKnownGoodCamera = cloneCamera(
        dom.embPlot?._fullLayout?.scene?.camera || layout.scene.camera || EMBED_3D_CAMERA_DEFAULT
      );
    } else {
      state.embeddings.lastKnownGoodCamera = null;
    }

    // Apply any active case-file / selection colors
    updateEmbeddingPlotColors();

    // Re-bind handlers safely when the plot is rebuilt.
    if (typeof dom.embPlot.removeAllListeners === 'function') {
      dom.embPlot.removeAllListeners('plotly_click');
      dom.embPlot.removeAllListeners('plotly_relayout');
      state.embeddings.plotHandlersBound = false;
    }

    if (!state.embeddings.plotHandlersBound) {
      // Click handler
      dom.embPlot.on('plotly_click', (eventData) => {
        // Ignore non-left interactions so panning/context-menu doesn't trigger selection logic.
        const evt = eventData?.event;
        if (evt) {
          if (evt.type === 'contextmenu') return;
          if (typeof evt.button === 'number' && evt.button !== 0) return;
          if (typeof evt.buttons === 'number' && evt.buttons !== 1) return;
        }

        if (eventData.points && eventData.points.length > 0) {
          const point = eventData.points[0];
          const idx = point.pointIndex ?? point.pointNumber;
          if (Number.isInteger(idx) && projData.words[idx]) {
            const word = projData.words[idx].word;
            if (state.embeddings.pendingSelectionTimer) {
              clearTimeout(state.embeddings.pendingSelectionTimer);
            }
            state.embeddings.pendingSelectionTimer = setTimeout(() => {
              state.embeddings.pendingSelectionTimer = null;
              if (state.embeddings.currentDim === '3d' && Plotly?.Fx?.unhover) {
                Plotly.Fx.unhover(dom.embPlot);
              }
              selectEmbeddingWord(word, { preserveViewport: true });
            }, 0);
          }
        }
      });

      // Guard against invalid camera states from over-zooming in 3D.
      dom.embPlot.on('plotly_relayout', (eventData) => {
        if (state.embeddings.currentDim !== '3d') return;
        if (state.embeddings.cameraCorrectionInFlight) return;

        const relayoutCamera = eventData && eventData['scene.camera'];
        const currentCamera = relayoutCamera ||
          dom.embPlot?._fullLayout?.scene?.camera ||
          dom.embPlot?.layout?.scene?.camera;
        const fallbackCamera = state.embeddings.lastKnownGoodCamera || EMBED_3D_CAMERA_DEFAULT;
        const { camera, changed, valid } = sanitize3dCamera(currentCamera, fallbackCamera);
        if (!valid || !camera) return;

        if (!changed) {
          state.embeddings.lastKnownGoodCamera = cloneCamera(camera);
          state.embeddings.lastCameraCorrectionSig = null;
          return;
        }

        const targetSig = cameraSignature(camera);
        if (state.embeddings.lastCameraCorrectionSig === targetSig) return;

        state.embeddings.cameraCorrectionInFlight = true;
        state.embeddings.lastCameraCorrectionSig = targetSig;
        Promise.resolve(Plotly.relayout(dom.embPlot, { 'scene.camera': camera }))
          .catch((err) => {
            console.warn('Embedding camera correction failed:', err);
          })
          .finally(() => {
            state.embeddings.lastKnownGoodCamera = cloneCamera(camera);
            state.embeddings.cameraCorrectionInFlight = false;
          });
      });

      state.embeddings.plotHandlersBound = true;
    }
  }

  function bindEmbeddingMouseGuards() {
    if (state.embeddings.mouseGuardsBound) return;

    // Plotly 2D right-button drag can throw internal relayout errors.
    // Block right-button drag in 2D only; keep 3D right-click pan intact.
    dom.embPlot.addEventListener('mousedown', (event) => {
      if (state.embeddings.currentDim === '2d' && event.button === 2) {
        event.preventDefault();
        event.stopPropagation();
      }
    }, true);

    dom.embPlot.addEventListener('contextmenu', (event) => {
      if (state.embeddings.currentDim === '2d') {
        event.preventDefault();
      }
    });

    state.embeddings.mouseGuardsBound = true;
  }

  function updateEmbeddingPlotColors() {
    const projData = state.data.projected;
    const words = projData.words;
    const categories = projData.categories;
    const active = state.embeddings.activeCaseFiles;
    const selected = state.embeddings.selectedWord;
    const is3d = state.embeddings.currentDim === '3d';

    const colors = [...state.embeddings.baseColors];
    const sizes  = [...state.embeddings.baseSizes];

    // Color active case files
    active.forEach(catId => {
      const catColor = categories[catId]?.color || '#999';
      words.forEach((w, i) => {
        if (w.categories.includes(catId)) {
          colors[i] = catColor;
          sizes[i] = CATEGORY_DOT_SIZE;
        }
      });
    });

    // Highlight selected word
    if (selected) {
      const idx = words.findIndex(w => w.word === selected);
      if (idx >= 0) {
        colors[idx] = SELECTED_DOT_COLOR;
        sizes[idx] = SELECTED_DOT_SIZE;
      }
    }

    // 3D scatter uses smaller marker sizes and doesn't support per-point opacity arrays
    const plotSizes = is3d ? sizes.map(s => s * 0.8) : sizes;
    const opacities = is3d ? 0.7 : colors.map(c => c === GENERAL_DOT_COLOR ? 0.4 : 0.9);

    swallowPlotlyRejection(Plotly.restyle(dom.embPlot, {
      'marker.color': [colors],
      'marker.size':  [plotSizes],
      'marker.opacity': [opacities],
    }, [0]), 'Embedding restyle failed');
  }

  function toggleCaseFile(catId) {
    const active = state.embeddings.activeCaseFiles;
    if (active.has(catId)) {
      active.delete(catId);
    } else {
      active.add(catId);
    }

    // Update button appearance
    dom.embCaseFiles.querySelectorAll('button').forEach(btn => {
      const isActive = active.has(btn.dataset.casefile);
      btn.classList.toggle('is-primary', isActive);
    });

    updateEmbeddingPlotColors();
  }

  function embeddingSearch() {
    const word = dom.embSearch.value.trim().toLowerCase();
    if (!word) return;

    const vectors = state.data.full.vectors;
    if (!vectors[word]) {
      show(dom.embOovNotice);
      return;
    }
    hide(dom.embOovNotice);
    selectEmbeddingWord(word);
  }

  function selectEmbeddingWord(word, options = {}) {
    const { preserveViewport = false } = options;
    state.embeddings.selectedWord = word;
    dom.embSelectedWord.textContent = `- "${word}"`;

    // Update plot
    if (state.embeddings.currentDim === '3d' && Plotly?.Fx?.unhover) {
      Plotly.Fx.unhover(dom.embPlot);
    }
    if (state.embeddings.currentDim === '3d') {
      requestAnimationFrame(() => updateEmbeddingPlotColors());
    } else {
      updateEmbeddingPlotColors();
    }

    // Zoom to word
    const projData = state.data.projected;
    const wData = projData.words.find(w => w.word === word);
    if (wData) {
      const is3d = state.embeddings.currentDim === '3d';
      if (is3d) {
        // Keep current 3D view stable; selection should not snap camera/zoom.
        const current = dom.embPlot?._fullLayout?.scene?.camera || dom.embPlot?.layout?.scene?.camera;
        if (current) state.embeddings.lastKnownGoodCamera = cloneCamera(current);
      } else {
        // Preserve user zoom/pan for click selection in 2D.
        if (!preserveViewport) {
          const pad = 15;
          Plotly.relayout(dom.embPlot, {
            'xaxis.range': [wData.x - pad, wData.x + pad],
            'yaxis.range': [wData.y - pad, wData.y + pad],
          });
        }
      }
    }

    // Show neighbors
    const vectors = state.data.full.vectors;
    if (vectors[word]) {
      const neighbors = findNeighbors(vectors[word], NEIGHBOR_COUNT, new Set([word]));
      renderNeighborList(dom.embNeighborsList, neighbors);
    }
  }

  function embeddingReset() {
    state.embeddings.selectedWord = null;
    state.embeddings.activeCaseFiles.clear();
    dom.embSelectedWord.textContent = '';
    dom.embSearch.value = '';
    hide(dom.embOovNotice);
    dom.embNeighborsList.innerHTML = '<li class="has-text-grey-light is-italic">Search or click a word to see neighbors</li>';
    dom.embAnalogyResult.textContent = '';
    hide(dom.embAnalogyNbrs);

    // Reset button states
    dom.embCaseFiles.querySelectorAll('button').forEach(btn => {
      btn.classList.remove('is-primary');
    });

    // Reset plot colors and zoom
    updateEmbeddingPlotColors();
    const is3d = state.embeddings.currentDim === '3d';
    if (is3d) {
      state.embeddings.lastKnownGoodCamera = cloneCamera(EMBED_3D_CAMERA_DEFAULT);
      Plotly.relayout(dom.embPlot, {
        'scene.camera': { ...EMBED_3D_CAMERA_DEFAULT },
      });
    } else {
      Plotly.relayout(dom.embPlot, {
        'xaxis.autorange': true,
        'yaxis.autorange': true,
      });
    }
  }

  function computeAnalogy() {
    const a = dom.embAnalogyA.value.trim().toLowerCase();
    const b = dom.embAnalogyB.value.trim().toLowerCase();
    const c = dom.embAnalogyC.value.trim().toLowerCase();
    const vectors = state.data.full.vectors;

    if (!a || !b || !c) return;

    const missing = [a, b, c].filter(w => !vectors[w]);
    if (missing.length > 0) {
      dom.embAnalogyResult.textContent = `Not in vocabulary: ${missing.join(', ')}`;
      dom.embAnalogyResult.className = 'has-text-danger';
      hide(dom.embAnalogyNbrs);
      return;
    }

    const resultVec = vectorAdd(vectorSub(vectors[a], vectors[b]), vectors[c]);
    const neighbors = findNeighbors(resultVec, ANALOGY_RESULT_COUNT, new Set([a, b, c]));

    if (neighbors.length > 0) {
      const top = neighbors[0];
      dom.embAnalogyResult.textContent = `${top.word} (${top.similarity.toFixed(3)})`;
      dom.embAnalogyResult.className = 'has-text-weight-bold has-text-success';

      // Show full list
      renderNeighborList(dom.embAnalogyNbrsList, neighbors);
      show(dom.embAnalogyNbrs);

      // Highlight the result on the plot
      selectEmbeddingWord(top.word);
    }
  }


  // ============================================================
  // 10. ATTENTION VIEW
  // ============================================================

  function initAttention() {
    const data = state.data.attention;
    const meta = data.metadata;

    // Populate sentence selector
    clearChildren(dom.attSentenceSelect);
    Object.entries(data.sentences).forEach(([id, sent]) => {
      const opt = document.createElement('option');
      opt.value = id;
      const displayText = sent.text.length > 70 ? sent.text.slice(0, 67) + '…' : sent.text;
      opt.textContent = displayText;
      dom.attSentenceSelect.appendChild(opt);
    });

    // Populate layer selector
    clearChildren(dom.attLayerSelect);
    for (let i = 0; i < meta.nLayers; i++) {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = i;
      dom.attLayerSelect.appendChild(opt);
    }

    // Populate head selector
    clearChildren(dom.attHeadSelect);
    for (let i = 0; i < meta.nHeads; i++) {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = i;
      dom.attHeadSelect.appendChild(opt);
    }

    // Set layer slider max
    dom.attLayerSlider.max = meta.nLayers; // 0 = embedding, 1..12 = encoder layers

    // Event listeners
    dom.attSentenceSelect.addEventListener('change', () => {
      loadAttentionSentence(dom.attSentenceSelect.value);
    });
    dom.attLayerSelect.addEventListener('change', () => {
      state.attention.currentLayer = parseInt(dom.attLayerSelect.value);
      updateAttentionHeatmap();
      updateQKVDisplay();
    });
    dom.attHeadSelect.addEventListener('change', () => {
      state.attention.currentHead = parseInt(dom.attHeadSelect.value);
      updateAttentionHeatmap();
      updateQKVDisplay();
    });
    dom.attBestBtn.addEventListener('click', selectBestHead);
    dom.attTemp.addEventListener('input', () => {
      state.attention.temperature = parseFloat(dom.attTemp.value);
      dom.attTempValue.textContent = state.attention.temperature.toFixed(1);
      updateAttentionHeatmap();
    });
    dom.attQkvToggle.addEventListener('click', toggleQKV);
    dom.attLayerSlider.addEventListener('input', () => {
      state.attention.projectionLayer = parseInt(dom.attLayerSlider.value);
      dom.attLayerSliderVal.textContent = state.attention.projectionLayer;
      updateAttentionProjection();
    });
    dom.attPlayBtn.addEventListener('click', playProjectionAnimation);

    // Load first sentence and apply best head
    const firstId = Object.keys(data.sentences)[0];
    loadAttentionSentence(firstId);
    selectBestHead();
  }

  function loadAttentionSentence(sentId) {
    state.attention.currentSentenceId = sentId;
    state.attention.selectedTokenIndex = null;

    const data = state.data.attention;
    const sent = data.sentences[sentId];
    const best = data.bestDefaults[sentId];

    // Render token chips
    clearChildren(dom.attTokens);
    sent.tokens.forEach((tok, i) => {
      const span = document.createElement('span');
      span.className = 'tag is-medium';
      span.dataset.tokenIndex = i;
      span.style.margin = '2px';
      span.style.cursor = 'pointer';

      const isSpecial = sent.specialTokenMask[i];
      const isToi = sent.tokensOfInterest.some(t => t.index === i);

      if (isSpecial) {
        span.style.opacity = '0.4';
        span.style.fontStyle = 'italic';
      }
      if (isToi) {
        span.classList.add('is-warning');
        span.style.fontWeight = 'bold';
      }

      span.textContent = tok;
      span.addEventListener('click', () => selectAttentionToken(i));
      dom.attTokens.appendChild(span);
    });

    // Set default layer/head to best
    dom.attLayerSelect.value = best.layer;
    dom.attHeadSelect.value = best.head;
    state.attention.currentLayer = best.layer;
    state.attention.currentHead = best.head;

    // Reset neighbor panels
    dom.attStaticNbrs.innerHTML = '<li class="has-text-grey-light is-italic">Click a token above</li>';
    dom.attContextualNbrs.innerHTML = '<li class="has-text-grey-light is-italic">Click a token above</li>';
    dom.attSelectedToken.textContent = '';

    // Render heatmap + projection
    updateAttentionHeatmap();
    renderAttentionProjection();
  }

  function selectBestHead() {
    const sentId = state.attention.currentSentenceId;
    if (!sentId) return;
    const best = state.data.attention.bestDefaults[sentId];
    dom.attLayerSelect.value = best.layer;
    dom.attHeadSelect.value = best.head;
    state.attention.currentLayer = best.layer;
    state.attention.currentHead = best.head;
    updateAttentionHeatmap();
    updateQKVDisplay();
  }

  function selectAttentionToken(tokenIdx) {
    state.attention.selectedTokenIndex = tokenIdx;
    const sentId = state.attention.currentSentenceId;
    const sent = state.data.attention.sentences[sentId];
    const layer = state.attention.currentLayer;

    // Highlight token chip
    dom.attTokens.querySelectorAll('.tag').forEach(span => {
      span.classList.remove('is-dark');
      if (parseInt(span.dataset.tokenIndex) === tokenIdx) {
        span.classList.add('is-dark');
      }
    });

    dom.attSelectedToken.textContent = `- "${sent.tokens[tokenIdx]}" (index ${tokenIdx})`;

    // Update heatmap row highlight
    highlightHeatmapRow(tokenIdx);

    // Update neighbor panels (only for tokens of interest)
    const toi = sent.tokensOfInterest.find(t => t.index === tokenIdx);
    if (toi) {
      renderNeighborList(dom.attStaticNbrs, toi.staticNeighbors.slice(0, NEIGHBOR_COUNT));
      const ctxKey = String(layer + 1); // hidden_states offset: layer 0 attention -> hidden_states[1]
      const ctxNeighbors = toi.contextualNeighbors[ctxKey] || toi.contextualNeighbors[String(layer)];
      renderNeighborList(dom.attContextualNbrs, ctxNeighbors ? ctxNeighbors.slice(0, NEIGHBOR_COUNT) : []);
    } else {
      dom.attStaticNbrs.innerHTML = '<li class="has-text-grey is-size-7">Neighbors pre-computed for highlighted tokens only</li>';
      dom.attContextualNbrs.innerHTML = '<li class="has-text-grey is-size-7">Neighbors pre-computed for highlighted tokens only</li>';
    }

    // Update QKV if visible
    if (state.attention.qkvVisible) {
      updateQKVDisplay();
    }
  }

  // Attention heatmap (D3)

  function updateAttentionHeatmap() {
    const sentId = state.attention.currentSentenceId;
    if (!sentId) return;

    const data = state.data.attention;
    const sent = data.sentences[sentId];
    const layer = state.attention.currentLayer;
    const head  = state.attention.currentHead;
    const temp  = state.attention.temperature;

    // Compute weights: reapply softmax with temperature if temp !== 1
    let weights;
    if (Math.abs(temp - 1.0) < 0.01) {
      weights = sent.attentionWeights[layer][head];
    } else {
      const rawScores = sent.qkScores[layer][head];
      weights = rawScores.map(row => softmax(row.map(s => s / temp)));
    }

    renderHeatmap(sent.tokens, weights, sent.qkScores[layer][head], sent.specialTokenMask);

    // Update neighbors for selected token if there is one
    if (state.attention.selectedTokenIndex !== null) {
      highlightHeatmapRow(state.attention.selectedTokenIndex);

      const toi = sent.tokensOfInterest.find(t => t.index === state.attention.selectedTokenIndex);
      if (toi) {
        const ctxKey = String(layer + 1);
        const ctxNeighbors = toi.contextualNeighbors[ctxKey] || toi.contextualNeighbors[String(layer)];
        renderNeighborList(dom.attContextualNbrs, ctxNeighbors ? ctxNeighbors.slice(0, NEIGHBOR_COUNT) : []);
      }
    }
  }

  function renderHeatmap(tokens, weights, rawScores, specialMask) {
    const container = dom.attHeatmap;
    clearChildren(container);

    const n = tokens.length;
    const margin = { top: 5, right: 5, bottom: 70, left: 70 };
    const availW = container.clientWidth || 450;
    const cellSize = Math.min(Math.floor((availW - margin.left - margin.right) / n), 36);
    const gridSize = cellSize * n;
    const width  = gridSize + margin.left + margin.right;
    const height = gridSize + margin.top + margin.bottom;

    const svg = d3.select(container).append('svg')
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const maxW = d3.max(weights.flat());
    const colorScale = d3.scaleSequential(HEATMAP_COLOR_SCALE).domain([0, maxW]);

    // Cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const w = weights[i][j];
        const cell = g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', colorScale(w))
          .attr('rx', 2)
          .attr('class', `hm-cell hm-row-${i}`)
          .attr('data-row', i)
          .attr('data-col', j);

        cell.on('mouseenter', function (event) {
          const score = rawScores[i][j];
          showTooltip(
            event.clientX, event.clientY,
            `<strong>Q</strong>(${tokens[i]}) · <strong>K</strong>(${tokens[j]})<br>` +
            `Score: ${score.toFixed(2)}<br>Weight: ${w.toFixed(4)}`
          );
        });
        cell.on('mouseleave', hideTooltip);
        cell.on('click', () => selectAttentionToken(i));
      }
    }

    // Y-axis labels (query / row)
    tokens.forEach((tok, i) => {
      g.append('text')
        .attr('x', -4)
        .attr('y', i * cellSize + cellSize / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('font-size', Math.min(cellSize - 4, 11) + 'px')
        .attr('fill', specialMask[i] ? '#CCC' : '#333')
        .text(tok);
    });

    // X-axis labels (key / column)
    tokens.forEach((tok, j) => {
      g.append('text')
        .attr('x', j * cellSize + cellSize / 2)
        .attr('y', gridSize + 10)
        .attr('text-anchor', 'start')
        .attr('font-size', Math.min(cellSize - 4, 11) + 'px')
        .attr('fill', specialMask[j] ? '#CCC' : '#333')
        .attr('transform', `rotate(45, ${j * cellSize + cellSize / 2}, ${gridSize + 10})`)
        .text(tok);
    });

    state.attention.heatmapRendered = true;
  }

  function highlightHeatmapRow(rowIdx) {
    if (!state.attention.heatmapRendered) return;
    // Dim all cells, brighten selected row
    d3.select(dom.attHeatmap).selectAll('.hm-cell')
      .attr('stroke', function () {
        return parseInt(this.dataset.row) === rowIdx ? '#333' : 'none';
      })
      .attr('stroke-width', function () {
        return parseInt(this.dataset.row) === rowIdx ? 1.5 : 0;
      });
  }

  // Q/K/V vector display

  function toggleQKV() {
    state.attention.qkvVisible = !state.attention.qkvVisible;
    dom.attQkvToggle.textContent = state.attention.qkvVisible
      ? 'Hide Q / K / V Vectors'
      : 'Show Q / K / V Vectors';
    dom.attQkvToggle.dataset.active = state.attention.qkvVisible;

    if (state.attention.qkvVisible) {
      show(dom.attQkvDisplay);
      updateQKVDisplay();
    } else {
      hide(dom.attQkvDisplay);
    }
  }

  function updateQKVDisplay() {
    if (!state.attention.qkvVisible) return;

    const sentId = state.attention.currentSentenceId;
    const sent = state.data.attention.sentences[sentId];
    const layer = state.attention.currentLayer;
    const head  = state.attention.currentHead;
    const tokIdx = state.attention.selectedTokenIndex;

    // Check if QKV data is available for this layer
    const qkvData = sent.qkv[String(layer)];
    if (!qkvData) {
      const available = Object.keys(sent.qkv).join(', ');
      [dom.attQkvQ, dom.attQkvK, dom.attQkvV].forEach(el => {
        el.innerHTML = `<p class="is-size-7 has-text-grey">QKV data exported for layers: ${available}</p>`;
      });
      return;
    }

    if (tokIdx === null) {
      [dom.attQkvQ, dom.attQkvK, dom.attQkvV].forEach(el => {
        el.innerHTML = '<p class="is-size-7 has-text-grey">Click a token to see its Q/K/V vectors</p>';
      });
      return;
    }

    const q = qkvData.q[head][tokIdx];
    const k = qkvData.k[head][tokIdx];
    const v = qkvData.v[head][tokIdx];

    renderVectorStrip(dom.attQkvQ, q, QKV_COLORS.q);
    renderVectorStrip(dom.attQkvK, k, QKV_COLORS.k);
    renderVectorStrip(dom.attQkvV, v, QKV_COLORS.v);
  }

  function renderVectorStrip(container, vector, accentColor) {
    clearChildren(container);
    const width = container.clientWidth || 250;
    const height = 30;
    const cellW = width / vector.length;

    const svg = d3.select(container).append('svg')
      .attr('width', width)
      .attr('height', height);

    const absMax = d3.max(vector.map(Math.abs)) || 1;
    const colorScale = d3.scaleDiverging()
      .domain([-absMax, 0, absMax])
      .interpolator(d3.interpolateRdBu);

    svg.selectAll('rect')
      .data(vector)
      .join('rect')
      .attr('x', (d, i) => i * cellW)
      .attr('y', 0)
      .attr('width', Math.max(cellW - 0.5, 0.5))
      .attr('height', height)
      .attr('fill', d => colorScale(d));

    // Subtle accent border
    svg.append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', width).attr('height', height)
      .attr('fill', 'none')
      .attr('stroke', accentColor)
      .attr('stroke-width', 1.5)
      .attr('rx', 2);
  }


  // Sentence projection (Plotly)

  function renderAttentionProjection() {
    const sentId = state.attention.currentSentenceId;
    if (!sentId) return;

    const sent = state.data.attention.sentences[sentId];
    const layerKey = String(state.attention.projectionLayer);
    const positions = sent.sentenceProjections[layerKey];
    if (!positions) return;

    const tokens = sent.tokens;
    const special = sent.specialTokenMask;
    const toi = sent.tokensOfInterest;

    const xs = [], ys = [], texts = [], colors = [], sizes = [];
    tokens.forEach((tok, i) => {
      if (special[i]) return; // skip [CLS], [SEP]
      xs.push(positions[i][0]);
      ys.push(positions[i][1]);
      texts.push(tok);

      const isToI = toi.some(t => t.index === i);
      colors.push(isToI ? '#E74C3C' : '#2980B9');
      sizes.push(isToI ? 14 : 8);
    });

    const trace = {
      x: xs, y: ys, text: texts,
      mode: 'markers+text',
      type: 'scatter',
      textposition: 'top center',
      textfont: { size: 10 },
      hoverinfo: 'text',
      marker: { color: colors, size: sizes, opacity: 0.8 },
    };

    const layout = {
      hovermode: 'closest',
      xaxis: { visible: false },
      yaxis: { visible: false, scaleanchor: 'x' },
      margin: { l: 10, r: 10, t: 10, b: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
    };

    const config = { responsive: true, displayModeBar: false };

    if (!state.attention.projectionInitialized) {
      Plotly.newPlot(dom.attProjection, [trace], layout, config);
      state.attention.projectionInitialized = true;
    } else {
      Plotly.react(dom.attProjection, [trace], layout, config);
    }
  }

  function updateAttentionProjection() {
    renderAttentionProjection();
  }

  function playProjectionAnimation() {
    // Stop existing animation
    if (state.attention.animationTimer) {
      clearInterval(state.attention.animationTimer);
      state.attention.animationTimer = null;
      dom.attPlayBtn.textContent = '▶ Play';
      return;
    }

    const maxLayer = parseInt(dom.attLayerSlider.max);
    state.attention.projectionLayer = 0;
    dom.attLayerSlider.value = 0;
    dom.attLayerSliderVal.textContent = '0';
    updateAttentionProjection();
    dom.attPlayBtn.textContent = '⏸ Pause';

    state.attention.animationTimer = setInterval(() => {
      state.attention.projectionLayer++;
      if (state.attention.projectionLayer > maxLayer) {
        clearInterval(state.attention.animationTimer);
        state.attention.animationTimer = null;
        dom.attPlayBtn.textContent = '▶ Play';
        return;
      }
      dom.attLayerSlider.value = state.attention.projectionLayer;
      dom.attLayerSliderVal.textContent = state.attention.projectionLayer;
      updateAttentionProjection();
    }, ANIMATION_STEP_MS);
  }


  // ============================================================
  // 11. MAIN INITIALIZATION
  // ============================================================

  try {
    await loadData();

    // Hide loader, show app
    hide(dom.appLoader);
    show(dom.app);

    // Bind tab navigation
    bindTabEvents();

    // Initialize the first visible view (tokenizer)
    switchView('tokenizer');

    console.log('Transformer Explorer initialized.');
    console.log(`  Tokenizer examples: ${state.data.tokenizer.examples.length}`);
    console.log(`  Embedding words: ${state.data.projected.words.length}`);
    console.log(`  Attention sentences: ${Object.keys(state.data.attention.sentences).length}`);
    console.log(`  Live tokenizer: ${liveTokenizer ? 'available' : 'not detected'}`);

  } catch (err) {
    console.error('Initialization failed:', err);
    dom.loaderStatus.textContent = 'Failed to initialize. Check console for details.';
    dom.loaderProgress.classList.add('is-danger');
  }

});
