(() => {
  'use strict';

  const local = (en, zh) => document.documentElement.lang.startsWith('zh') ? zh : en;
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function renderSummary() {
    const root = document.getElementById('results-summary');
    const table = document.getElementById('results-table');
    if (!root || !table) return;
    root.replaceChildren();
    root.setAttribute('aria-label', local('Results at a glance', '实验结果速览'));
    const columns = [...table.querySelectorAll('thead th')].slice(1).map(cell => cell.textContent.trim());
    const rows = [...table.querySelectorAll('tbody tr')].map(row => ({
      name: row.querySelector('th')?.textContent.trim() || '',
      ours: row.classList.contains('ours'),
      cells: [...row.querySelectorAll('td')].map(cell => {
        const display = cell.textContent.trim();
        const value = display && display !== '—' ? Number(display) : NaN;
        return { display, value: Number.isFinite(value) ? value : null };
      })
    }));
    const ours = rows.filter(row => row.ours);
    const baselines = rows.filter(row => !row.ours);
    const strongestBaseline = index => baselines.filter(row => row.cells[index]?.value != null)
      .sort((a, b) => b.cells[index].value - a.cells[index].value)[0];

    if (columns.length === 6) {
      const index = columns.findIndex(column => /^SM\b|ScanMatch/i.test(column));
      const candidate = ours.find(row => /CheX/i.test(row.name));
      const baseline = index >= 0 ? strongestBaseline(index) : null;
      if (candidate?.cells[index]?.value !== null && candidate?.cells[index] && baseline) {
        const score = candidate.cells[index];
        const reference = baseline.cells[index];
        const difference = score.value - reference.value;
        const delta = `${difference >= 0 ? '+' : '−'}${Math.abs(difference).toFixed(3)}`;
        const note = element('p', 'comparison-note');
        const label = element('strong', '', local('ScanMatch ↑ · GazeWorld (CheX)', 'ScanMatch ↑ · GazeWorld（CheX）'));
        note.append(label, document.createTextNode(local(
          ` scores ${score.display}, compared with ${reference.display} for ${baseline.name}, the strongest reported baseline (${delta}). Values match the table below.`,
          ` 得分为 ${score.display}，最佳已报告基线 ${baseline.name} 为 ${reference.display}（${delta}）。数值与下方表格一致。`
        )));
        root.append(note);
      }
    } else if (columns.length === 3 && ours.length) {
      columns.forEach((column, index) => {
        const score = ours[0].cells[index];
        const baseline = strongestBaseline(index);
        if (!score || score.value === null || !baseline) return;
        const reference = baseline.cells[index];
        const difference = score.value - reference.value;
        const card = element('article', 'comparison-card');
        const scoreline = element('div', 'comparison-scoreline');
        const value = element('strong', 'comparison-score', score.display);
        value.append(element('small', '', '%'));
        const signed = `${difference >= 0 ? '+' : '−'}${Math.abs(difference).toFixed(2)}`;
        const delta = element('span', `comparison-delta${difference < 0 ? ' negative' : ''}`, local(
          `${signed} pts vs. best baseline`,
          `较最佳基线 ${signed} 个百分点`
        ));
        scoreline.append(value, delta);
        card.append(element('h4', 'comparison-title', column), scoreline);
        [
          { kind: 'ours', label: 'GazeWorld', cell: score },
          { kind: 'baseline', label: local(`${baseline.name} · baseline`, `${baseline.name} · 基线`), cell: reference }
        ].forEach(series => {
          const row = element('div', `comparison-series ${series.kind}`);
          const label = element('div', 'comparison-bar-label');
          label.append(element('span', '', series.label), element('span', '', series.cell.display));
          const track = element('div', 'comparison-track');
          track.setAttribute('aria-hidden', 'true');
          const bar = element('span', 'comparison-bar');
          bar.style.width = `${Math.min(100, Math.max(0, series.cell.value))}%`;
          track.append(bar);
          row.append(label, track);
          card.append(row);
        });
        root.append(card);
      });
    }
    root.hidden = !root.childElementCount;
  }

  document.addEventListener('gazeworld:results', renderSummary);
  renderSummary();

  // Keep figure links functional when native dialogs are unavailable.
  if (typeof HTMLDialogElement === 'undefined' || !HTMLDialogElement.prototype.showModal) return;

  let dialog;
  let image;
  let title;
  let caption;
  let original;
  let close;
  let opener;
  let previousOverflow = '';

  function createDialog() {
    dialog = element('dialog', 'figure-dialog');
    dialog.setAttribute('aria-labelledby', 'figure-dialog-title');
    const toolbar = element('div', 'figure-dialog-toolbar');
    title = element('h2', 'figure-dialog-title');
    title.id = 'figure-dialog-title';
    const actions = element('div', 'figure-dialog-actions');
    original = element('a', 'figure-original');
    original.target = '_blank';
    original.rel = 'noopener noreferrer';
    close = element('button', 'figure-close');
    close.type = 'button';
    close.autofocus = true;
    actions.append(original, close);
    toolbar.append(title, actions);
    const canvas = element('div', 'figure-dialog-canvas');
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'region');
    image = element('img');
    image.decoding = 'async';
    canvas.append(image);
    caption = element('p', 'figure-dialog-caption');
    dialog.append(toolbar, canvas, caption);
    document.body.append(dialog);
    close.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    });
    dialog.addEventListener('close', () => {
      document.documentElement.style.overflow = previousOverflow;
      document.documentElement.classList.remove('figure-viewer-open');
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    });
  }

  document.addEventListener('click', event => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.('.paper-figure > a, .paper-figure a.figure-image, .paper-figure a.figure-open');
    const figure = link?.closest('.paper-figure');
    const sourceImage = figure?.querySelector('img');
    if (!sourceImage) return;
    if (!dialog) createDialog();
    title.textContent = figure.querySelector('.figure-heading h3')?.textContent || local('GazeWorld · Research figure', 'GazeWorld · 研究图示');
    dialog.querySelector('.figure-dialog-canvas').setAttribute('aria-label', local('Research figure. Scroll to explore.', '研究图示，可滚动查看。'));
    image.src = link.href;
    image.alt = sourceImage.alt;
    caption.textContent = figure.querySelector('figcaption')?.textContent.trim() || sourceImage.alt;
    original.href = link.href;
    original.textContent = local('Original image ↗', '查看原图 ↗');
    close.textContent = '×';
    close.setAttribute('aria-label', local('Close image viewer', '关闭图片查看器'));
    try {
      dialog.showModal();
    } catch (_) {
      return;
    }
    event.preventDefault();
    opener = link;
    previousOverflow = document.documentElement.style.overflow;
    document.documentElement.style.overflow = 'hidden';
    document.documentElement.classList.add('figure-viewer-open');
  });
})();
