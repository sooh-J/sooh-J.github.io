(function () {
  'use strict';

  /* ---------- theme toggle ---------- */
  var root = document.documentElement;
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* ignore */ }
    });
  }

  /* ---------- timeline jumps: flash the Experience/Publication entry on arrival ---------- */
  document.querySelectorAll('[data-jump]').forEach(function (a) {
    a.addEventListener('click', function () {
      var target = document.getElementById(a.getAttribute('href').slice(1));
      // Flash single entries only; a whole section lighting up is too much
      if (!target || target.classList.contains('section')) return;
      // Restart the animation even when the same entry is clicked twice
      target.classList.remove('is-flash');
      void target.offsetWidth;
      target.classList.add('is-flash');
    });
  });

  document.addEventListener('animationend', function (e) {
    if (e.animationName === 'jump-flash') e.target.classList.remove('is-flash');
  });

  /* ---------- scrollspy: highlight nav link of the section in view ---------- */
  var navLinks = document.querySelectorAll('[data-nav]');
  if (!navLinks.length) return;

  var sections = [];
  navLinks.forEach(function (a) {
    var href = a.getAttribute('href');
    if (href && href.indexOf('#') === 0) {
      var el = document.querySelector(href);
      if (el) sections.push({ id: href, el: el, link: a });
    }
  });

  if (!sections.length) return;

  var topbar = document.getElementById('topbar');
  var topbarH = topbar ? topbar.offsetHeight : 0;

  function setActive(id) {
    sections.forEach(function (s) {
      if (s.id === id) s.link.classList.add('is-active');
      else s.link.classList.remove('is-active');
    });
  }

  function onScroll() {
    var pos = window.scrollY + topbarH + 32;
    var current = sections[0].id;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].el.offsetTop <= pos) {
        current = sections[i].id;
      } else {
        break;
      }
    }
    setActive(current);
  }

  var ticking = false;
  window.addEventListener('scroll', function () {
    if (!ticking) {
      window.requestAnimationFrame(function () {
        onScroll();
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });

  window.addEventListener('resize', function () {
    topbarH = topbar ? topbar.offsetHeight : 0;
    onScroll();
  });

  onScroll();
})();
