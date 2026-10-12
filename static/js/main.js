// Rotate any gauge needle to match its data-score attribute (0 - 1 scale)
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-gauge-score]').forEach((gauge) => {
    const score = parseFloat(gauge.getAttribute('data-gauge-score'));
    const needle = gauge.querySelector('.gauge-needle');
    if (!needle) return;
    const clamped = Math.max(0, Math.min(1, score));
    const angle = -90 + clamped * 180; // -90deg (0) to +90deg (1)
    requestAnimationFrame(() => {
      needle.style.transform = `rotate(${angle}deg)`;
    });
  });

  // Keep number inputs and their paired range sliders in sync
  document.querySelectorAll('[data-pair]').forEach((el) => {
    const pairId = el.getAttribute('data-pair');
    const partner = document.getElementById(pairId);
    if (!partner) return;
    el.addEventListener('input', () => {
      partner.value = el.value;
      const display = document.getElementById(pairId + '-display');
      if (display) display.textContent = el.value;
    });
  });
});
