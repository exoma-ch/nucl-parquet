<script>
  import Plotly from 'plotly.js-dist-min';

  let { z, a } = $props();

  let plotEl;
  let data = $state(null);
  let loading = $state(false);

  const TYPE_COLORS = {
    electron: '#6d9eeb', alpha: '#ef5350', photon: '#ab47bc',
  };

  const BIO_SCALES = [
    [0.001, 0.02, 'DNA'],
    [0.02, 4, 'Nucleus'],
    [4, 30, 'Cell'],
    [30, 500, 'Cluster'],
    [500, 1e4, 'Macro'],
    [1e4, 2e5, 'Bulk'],
  ];

  async function load() {
    loading = true;
    const res = await fetch(`/api/dose-kernel/${z}/${a}`);
    data = await res.json();
    loading = false;
    render();
  }

  function render() {
    if (!plotEl || !data || !data.r_um?.length) return;
    const traces = [];

    // Per-type shell energy traces (left y-axis)
    for (const [type, vals] of Object.entries(data.by_type)) {
      traces.push({
        x: data.r_um, y: vals,
        mode: 'lines', name: `${type} (shell)`,
        line: { color: TYPE_COLORS[type] || '#78909c', width: 1.5 },
        yaxis: 'y',
      });
    }

    // Total shell energy (left y-axis)
    traces.push({
      x: data.r_um, y: data.shell_MeV_per_um,
      mode: 'lines', name: 'Total (shell)',
      line: { color: '#e8eaed', width: 2.5 },
      yaxis: 'y',
    });

    // Cumulative F(r) (right y-axis)
    traces.push({
      x: data.r_um, y: data.cumulative_F,
      mode: 'lines', name: 'F(r) cumulative',
      line: { color: '#ffa726', width: 2, dash: 'dot' },
      yaxis: 'y2',
    });

    // Bio-scale band shapes + annotations
    const shapes = [];
    const annotations = [];
    for (const [x0, x1, label] of BIO_SCALES) {
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0, x1, y0: 0, y1: 1,
        fillcolor: 'rgba(255,255,255,0.03)', line: { width: 0 },
      });
      annotations.push({
        x: Math.log10(Math.sqrt(x0 * x1)), y: 1.06,
        xref: 'x', yref: 'paper',
        text: label, showarrow: false,
        font: { size: 9, color: '#555' },
      });
    }

    // R50 / R90 vertical lines
    if (data.r50_um) {
      shapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: data.r50_um, x1: data.r50_um, y0: 0, y1: 1,
        line: { color: '#93c47d', width: 1, dash: 'dash' },
      });
      annotations.push({
        x: Math.log10(data.r50_um), y: 0.52, xref: 'x', yref: 'y2 domain',
        text: `R₅₀`, showarrow: true, arrowhead: 0, ax: 20, ay: 0,
        font: { size: 10, color: '#93c47d' },
      });
    }
    if (data.r90_um) {
      shapes.push({
        type: 'line', xref: 'x', yref: 'paper',
        x0: data.r90_um, x1: data.r90_um, y0: 0, y1: 1,
        line: { color: '#ffa726', width: 1, dash: 'dash' },
      });
      annotations.push({
        x: Math.log10(data.r90_um), y: 0.92, xref: 'x', yref: 'y2 domain',
        text: `R₉₀`, showarrow: true, arrowhead: 0, ax: 20, ay: 0,
        font: { size: 10, color: '#ffa726' },
      });
    }

    Plotly.react(plotEl, traces, {
      xaxis: {
        type: 'log', title: 'Distance (μm)',
        gridcolor: '#2a2d38', zerolinecolor: '#2a2d38',
      },
      yaxis: {
        type: 'log', title: '4πr²ρ·D(r) [MeV/μm/decay]',
        titlefont: { color: '#c9cdd5', size: 11 },
        gridcolor: '#2a2d38', zerolinecolor: '#2a2d38',
        side: 'left',
      },
      yaxis2: {
        title: 'F(r) — cumulative fraction',
        titlefont: { color: '#ffa726', size: 11 },
        overlaying: 'y', side: 'right',
        range: [0, 1.05], showgrid: false,
        tickvals: [0, 0.25, 0.5, 0.75, 0.9, 1.0],
        tickfont: { color: '#ffa726' },
      },
      shapes, annotations,
      plot_bgcolor: '#1a1c25', paper_bgcolor: '#1a1c25',
      font: { color: '#c9cdd5', size: 11 },
      legend: { x: 0.01, y: 0.95, bgcolor: 'rgba(26,28,37,0.9)', bordercolor: '#2a2d38' },
      margin: { t: 10, r: 55, b: 40, l: 60 },
      height: 360,
    }, { responsive: true });
  }

  function fmtR(v) {
    if (v == null) return '—';
    if (v < 0.1) return `${(v * 1000).toFixed(0)} nm`;
    if (v < 1000) return `${v.toFixed(1)} μm`;
    if (v < 1e4) return `${(v / 1000).toFixed(1)} mm`;
    return `${(v / 1e4).toFixed(1)} cm`;
  }

  $effect(() => {
    void z; void a;
    if (z && a) load();
  });
</script>

<div class="dose-kernel">
  {#if loading}
    <p class="loading">Computing dose kernel...</p>
  {:else if data && data.r_um?.length}
    <div class="dk-summary">
      <span>R₅₀ = <strong>{fmtR(data.r50_um)}</strong></span>
      <span>R₉₀ = <strong>{fmtR(data.r90_um)}</strong></span>
      <span>E<sub>tot</sub> = <strong>{data.total_energy_MeV?.toFixed(3)}</strong> MeV/decay</span>
    </div>
  {:else if data}
    <p style="color: var(--text-dim); font-size: 12px;">No emission data for dose kernel</p>
  {/if}
  <div bind:this={plotEl}></div>
</div>

<style>
  .dose-kernel {
    background: var(--bg2); border-radius: 6px;
    border: 1px solid var(--border); padding: 8px; margin-top: 8px;
  }
  .dk-summary {
    display: flex; gap: 16px; font-size: 12px; color: var(--text);
    font-family: var(--mono); padding: 4px 0 6px;
  }
  .dk-summary strong { color: #e8eaed; }
</style>
