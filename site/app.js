(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const translations = {
    skip: '跳转到正文', navAbout: '概览', navAbstract: '摘要', navDemo: '演示', navGallery: '图例', navCitation: '引用', navMethod: '方法', navResults: '实验结果', navResources: '资源',
    heroTitle: '放射科医生阅片的<br>世界模型', paperSubtitle: '面向医学影像表征学习',
    abstractTitle: '摘要',
    abstractText1: '放射科医生的眼动记录了搜索、比较与积累证据的过程。GazeWorld 将胸片视为感知世界，将医生的注视序列视为其中的探索轨迹。自回归预测器根据已访问图像块，预测下一个被注视图像块的潜在表征；空间补全分支则预测未访问区域的表征。',
    abstractText2: '推理时，模型仅需图像，无需注视数据。学习到的冻结特征可用于 CheXpert、RSNA Pneumonia 与 SIIM-ACR Pneumothorax 上的诊断分类，以及 GazeSearch 上的扫描路径预测。这些结果表明，建模专家如何阅片能够为医学影像表征学习提供有效的预训练信号。',
    inferenceTitle: '仅用图像进行推理', inferenceText: '注视数据指导预训练。推理时，GazeWorld 仅使用图像，其冻结表征可迁移至诊断分类与扫描路径探测。',
    demoTitle: '交互示意：沿着阅片轨迹探索', demoIntro: '播放序列或拖动滑块，观察已访问图像块如何成为下一步预测的上下文。本示意用于说明方法，并非实时模型输出。',
    demoGuideTitle: '从注视到表征', demoStep1Title: '观察轨迹。', demoStep1: '每个编号代表已访问的图像块，访问顺序为模型提供序列上下文。', demoStep2Title: '预测下一图像块。', demoStep2: '自回归分支在潜在空间中预测下一次注视的表征。', demoStep3Title: '补全整个世界。', demoStep3: '空间补全目标将学习扩展至注视轨迹以外的区域。', demoNote: '此处展示的是示意序列。论文中的预测示例见下方图例。', galleryTitle: '扫描路径图例', footerSource: '本项目代码已开源。',
    eyebrow: '从专家视角，理解医学影像',
    publication: '开放研究 · arXiv 2026', heroTagline: '从<em>专家的目光</em>中学习。',
    topicVision: '医学影像', topicWorld: '世界模型', topicGaze: '专家注视',
    researchTeam: '研究团队', architectureTitle: '走进世界模型', expandFigure: '放大查看',
    fullComparison: '完整基准对比', scrollHint: '横向滑动查看 →', viewerIndex: '方法示意',
    paperTitle: 'GazeWorld：面向医学影像表征学习的放射科医生阅片世界模型',
    heroDescription: 'GazeWorld 从放射科医生的阅片过程学习：预测下一个被注视图像块，补全未访问区域，获得可迁移的医学影像表征。',
    readPaper: '论文', exploreMethod: '了解方法', authorNote: '* 共同第一作者 &nbsp; † 通讯作者',
    viewerTitle: '阅片轨迹', fixationStep: '注视步骤', fixationLegend: '已观察的注视点', illustrationNote: '示意轨迹，非模型预测',
    keyIdea: '核心思想', keyIdeaText: '学习<em>专家如何阅片</em>，理解<em>影像中的信息。</em>',
    overviewLabel: '01 / 研究概述', overviewTitle: '从关注哪里，<br><em>到预测下一步。</em>',
    overviewText: '专家的眼动记录了搜索、比较和积累证据的过程。GazeWorld 将这一过程转化为学习信号：预测下一个被注视图像块的表征，同时补全尚未探索区域的表征。',
    overviewText2: '预训练利用医生的注视数据，下游推理只需图像。冻结后的视觉特征可用于诊断分类与扫描路径预测。',
    fullPaper: '在 arXiv 阅读完整论文', metric1: '在所报告方法中取得最佳 AUROC', metric1Detail: '监督分类 · 3 个数据集 × 3 种标注比例',
    metric2: 'SIIM-ACR 分类 AUROC', metric2Detail: '冻结骨干网络 · 100% 标注', metric3: 'GazeSearch 上的 ScanMatch', metric3Detail: 'CheX 变体 · 冻结编码器',
    methodLabel: '02 / 方法设计', methodTitle: '30 秒了解方法', methodIntro: '两个互补的学习目标，将专家的阅片轨迹与完整影像联系起来。',
    stageInput: '感知', stagePredict: '预测', stageComplete: '补全',
    step1Title: '将图像视为一个世界', step1Text: 'MedCLIP-Swin 编码器将胸片转换为图像块表征，注视序列决定探索这个世界的顺序。',
    step2Title: '预测下一次注视', step2Text: '因果 Transformer 利用已访问图像块，在潜在空间中预测下一个被注视图像块的表征，EMA 编码器提供学习目标。',
    step3Title: '理解未访问的区域', step3Text: '空间补全分支预测未被注视区域的表征，让学习信号覆盖注视轨迹之外的影像内容。',
    objectiveLabel: '联合学习目标', objectiveDetail: 'λ = 1.0 · 潜在空间预测 · 推理无需注视数据',
    methodCaption: '论文图 1：GazeWorld 架构，包括下一注视预测、空间补全与下游任务。',
    resultsLabel: '03 / 实验结果', resultsTitle: '实验结果', resultsIntro: '论文报告的诊断分类与专家扫描路径预测结果。切换任务与标注比例，查看完整对比。',
    tabSupervised: '监督分类', tabZero: '零样本分类', tabScanpath: '扫描路径预测',
    resultsNote: '数值来自论文，并收录于项目仓库。— 表示未报告该数值。', viewSource: '查看来源 ↗',
    qualitativeTitle: '模型如何探索影像', qualitativeSubtitle: '人类注视 · GazeSearch · GazeWorld',
    scanpathCaption: '论文图 3：七种病理类别上的扫描路径定性对比。图中样例来自论文，非实时模型输出。',
    resourcesLabel: '04 / 开源研究', resourcesTitle: '代码与复现',
    resourcesIntro: '探索开源实现，复现实验，将专家阅片过程作为新的表征学习信号。', viewCode: '代码',
    resourceNote: '代码采用 MIT 许可证。数据集与预训练骨干权重需单独获取，并在运行前配置本地路径。',
    setupTab: '安装', pretrainTab: '预训练', evaluateTab: '评估', copy: '复制', dataGuide: '数据集获取与路径配置',
    citationLabel: '05 / 引用', citationTitle: '引用', citationIntro: '如果 GazeWorld 对你的研究有所帮助，欢迎引用我们的工作。',
    downloadBib: '下载 BibTeX', footerText: '从专家看待影像的方式中学习。'
  };
  const english = new Map($$('[data-i18n]').map(el => [el, el.innerHTML]));
  let lang = 'en';
  let task = 'supervised';
  let metric = 2;
  const announce = text => { $('#announcer').textContent = text; };
  const local = (en, zh) => lang === 'zh' ? zh : en;

  function setLanguage(next) {
    lang = next;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    $$('[data-i18n]').forEach(el => { el.innerHTML = lang === 'zh' ? (translations[el.dataset.i18n] || english.get(el)) : english.get(el); });
    $('#language').textContent = lang === 'zh' ? 'EN' : '中文';
    $('#language').setAttribute('aria-label', lang === 'zh' ? 'Switch to English' : '切换为中文');
    document.title = lang === 'zh' ? 'GazeWorld — 从专家视角理解医学影像' : 'GazeWorld — Learning from expert eyes';
    try { localStorage.setItem('gazeworld-language', lang); } catch (_) { /* Storage is optional. */ }
    renderResults();
    updatePlayLabel();
    renderGaze(Number($('#gaze-step').value));
  }
  $('#language').addEventListener('click', () => setLanguage(lang === 'en' ? 'zh' : 'en'));

  function renderResults() {
    const data = window.GAZEWORLD_RESULTS?.[task];
    if (!data) return;
    const titles = {
      supervised: ['Frozen-backbone linear probing', '冻结骨干网络的线性探测'],
      zero: ['Zero-shot diagnostic classification', '零样本诊断分类'],
      scanpath: ['GazeSearch scanpath probing', 'GazeSearch 扫描路径探测']
    };
    const labels = task === 'supervised' ? ['1%', '10%', '100%'] : ['AUROC', 'Acc', 'F1'];
    $('#results-title').textContent = local(...titles[task]);
    $('#results-description').textContent = task === 'scanpath' ? local('SM / STDE / Vec / Dir / Pos ↑ · SED ↓', 'SM / STDE / Vec / Dir / Pos 越高越好 · SED 越低越好') : `${task === 'supervised' ? 'AUROC' : labels[metric]} (%) · ${local('higher is better', '越高越好')}`;
    const options = $('#result-options');
    options.innerHTML = '';
    options.setAttribute('aria-label', task === 'supervised' ? local('Label fraction', '标注比例') : local('Evaluation metric', '评估指标'));
    if (task !== 'scanpath') labels.forEach((label, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = task === 'supervised' ? `${label} ${local('labels', '标注')}` : label;
      button.setAttribute('aria-pressed', String(metric === i));
      button.addEventListener('click', () => { metric = i; renderResults(); $('#result-options').children[i].focus(); });
      options.append(button);
    });
    const columns = data.columns;
    $('#results-table caption').textContent = `${local(...titles[task])}: ${task === 'scanpath' ? local('all metrics', '全部指标') : labels[metric]}`;
    $('#results-table thead').innerHTML = `<tr><th scope="col">${local('Method', '方法')}</th>${columns.map(c => `<th scope="col">${c}</th>`).join('')}</tr>`;
    const rows = data.rows.map(row => ({...row, displayed: task === 'scanpath' ? row.values : row.values.map(v => v[metric])}));
    const best = columns.map((_, i) => {
      const values = rows.map(r => r.displayed[i]).filter(v => v !== null);
      return task === 'scanpath' && i === 1 ? Math.min(...values) : Math.max(...values);
    });
    $('#results-table tbody').innerHTML = rows.map(row => `<tr${row.ours ? ' class="ours"' : ''}><th scope="row">${row.name}</th>${row.displayed.map((v, i) => `<td${v !== null && v === best[i] ? ' class="best"' : ''}>${v === null ? '—' : v.toFixed(task === 'scanpath' && i !== 1 ? 3 : 2)}</td>`).join('')}</tr>`).join('');
    document.dispatchEvent(new CustomEvent('gazeworld:results'));
  }
  function activateResult(button) {
    task = button.dataset.result;
    metric = task === 'supervised' ? 2 : 0;
    $$('[data-result]').forEach(tab => { const active = tab === button; tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1; });
    $('#results-panel').setAttribute('aria-labelledby', button.id);
    renderResults();
  }
  $$('[data-result]').forEach(button => button.addEventListener('click', () => activateResult(button)));
  function keyboardTabs(selector) {
    const tabs = $$(selector);
    tabs.forEach(tab => tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (tabs.indexOf(tab) + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (tabs.indexOf(tab) + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); tabs[next].focus(); tabs[next].click(); }
    }));
  }
  keyboardTabs('[data-result]');

  const commands = {
    setup: $('#command-text').textContent,
    pretrain: '# Configure dataset paths and OUT_PATH first.\n# Set OUT_PATH to outputs/fixation_cache.npz.\nmkdir -p outputs\npython scripts/preprocess_fixation_cache.py\n\npython scripts/train_v4b_stage1.py \\\n  --cache outputs/fixation_cache.npz \\\n  --out_dir outputs/gazeworld',
    evaluate: '# Requires CUDA, datasets, and a Stage-1 checkpoint.\npip install pydicom\n\npython scripts/benchmark_v4b.py \\\n  --stage1_ckpt outputs/gazeworld/stage1_ckpt.pth \\\n  --out outputs/gazeworld/linear_probe.json'
  };
  $$('[data-code]').forEach(button => button.addEventListener('click', () => {
    $$('[data-code]').forEach(tab => { const active = tab === button; tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1; });
    $('#command-text').textContent = commands[button.dataset.code];
    $('#code-content').setAttribute('aria-labelledby', button.id);
  }));
  keyboardTabs('[data-code]');
  async function copyText(text, button) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        const textarea = document.createElement('textarea');
        textarea.value = text; textarea.style.position = 'fixed'; textarea.style.opacity = '0';
        document.body.append(textarea); textarea.select();
        const success = document.execCommand('copy'); textarea.remove(); button.focus();
        if (!success) throw new Error('Copy unavailable');
      }
      button.textContent = local('Copied!', '已复制！');
      announce(local('Copied to clipboard.', '已复制到剪贴板。'));
    } catch (_) {
      button.textContent = local('Select text', '请选中文本');
      const target = button.id === 'copy-code' ? $('#command-text') : $('#citation-text');
      const range = document.createRange(); range.selectNodeContents(target);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      announce(local('Please copy the selected text manually.', '请手动复制选中的文本。'));
    }
    window.setTimeout(() => { button.textContent = local('Copy', '复制'); }, 2200);
  }
  $('#copy-code').addEventListener('click', event => copyText($('#command-text').textContent, event.currentTarget));
  $('#copy-citation').addEventListener('click', event => copyText($('#citation-text').textContent, event.currentTarget));

  const points = [[181,118],[329,155],[194,225],[346,270],[174,315],[277,297],[310,208]];
  let gazeTimer = null;
  const svgNS = 'http://www.w3.org/2000/svg';
  function svgElement(name, attributes) { const el = document.createElementNS(svgNS, name); Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, value)); return el; }
  function renderGaze(step) {
    const visited = points.slice(0, step);
    const path = $('#gaze-path'); const nodes = $('#gaze-points');
    path.replaceChildren(); nodes.replaceChildren();
    path.append(svgElement('polyline', {points: visited.map(p => p.join(',')).join(' '), class: 'gaze-line'}));
    visited.forEach(([x, y], i) => {
      if (i === step - 1) nodes.append(svgElement('circle', {cx:x, cy:y, r:25, class:'gaze-halo'}));
      nodes.append(svgElement('circle', {cx:x, cy:y, r:i === step - 1 ? 13 : 10, class:`gaze-node${i === step - 1 ? ' active' : ''}`}));
      const label = svgElement('text', {x, y, class:'gaze-number'}); label.textContent = i + 1; nodes.append(label);
    });
    const [x, y] = points[step - 1];
    $('#patch-highlight rect').setAttribute('x', Math.floor(x / (520 / 7)) * (520 / 7));
    $('#patch-highlight rect').setAttribute('y', Math.floor(y / (475 / 7)) * (475 / 7));
    $('#gaze-counter').innerHTML = `0${step} <span>/ 07</span>`;
    $('#gaze-step').value = step;
    $('#gaze-step').setAttribute('aria-valuetext', local(`Fixation ${step} of 7`, `第 ${step} 个注视点，共 7 个`));
  }
  function updatePlayLabel() {
    $('#play-icon').textContent = gazeTimer ? 'Ⅱ' : '▶';
    $('#play-gaze').setAttribute('aria-label', gazeTimer ? local('Pause illustrative gaze sequence', '暂停注视轨迹示意') : local('Play illustrative gaze sequence', '播放注视轨迹示意'));
  }
  function stopGaze() { if (gazeTimer) window.clearInterval(gazeTimer); gazeTimer = null; updatePlayLabel(); }
  $('#play-gaze').addEventListener('click', () => {
    if (gazeTimer) { stopGaze(); return; }
    if (Number($('#gaze-step').value) === 7) renderGaze(1);
    gazeTimer = window.setInterval(() => {
      const step = Number($('#gaze-step').value);
      if (step >= 7) { stopGaze(); return; }
      renderGaze(step + 1);
      if (step + 1 === 7) stopGaze();
    }, 800);
    updatePlayLabel();
  });
  $('#gaze-step').addEventListener('input', event => { stopGaze(); renderGaze(Number(event.target.value)); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopGaze(); });
  let savedLanguage;
  try { savedLanguage = localStorage.getItem('gazeworld-language'); } catch (_) { /* Use English without storage. */ }
  setLanguage(savedLanguage === 'zh' ? 'zh' : 'en');
  renderGaze(7);
})();
