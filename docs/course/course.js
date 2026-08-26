// TicketToRide RL Lab Course Engine JS
document.addEventListener('DOMContentLoaded', () => {
  // Theme Toggle Management
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const currentTheme = localStorage.getItem('ttr_course_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', currentTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const activeTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = activeTheme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('ttr_course_theme', newTheme);
      themeToggleBtn.innerHTML = newTheme === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode';
    });
  }

  // Quiz Solution Revealer
  document.querySelectorAll('.btn-reveal-solution').forEach(button => {
    button.addEventListener('click', (e) => {
      const targetId = button.getAttribute('data-target');
      const solutionBox = document.getElementById(targetId);
      if (solutionBox) {
        solutionBox.classList.toggle('visible');
        if (solutionBox.classList.contains('visible')) {
          button.textContent = 'Nascondi Spiegazione';
          button.style.background = 'var(--accent-green)';
          if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([solutionBox]);
          }
        } else {
          button.textContent = 'Verifica & Mostra Soluzione';
          button.style.background = 'var(--accent-blue)';
        }
      }
    });
  });

  // Track Lesson Completion
  const currentLessonMatch = window.location.pathname.match(/lesson_(\d+)/);
  if (currentLessonMatch) {
    const lessonNum = parseInt(currentLessonMatch[1], 10);
    let completedLessons = JSON.parse(localStorage.getItem('ttr_completed_lessons') || '[]');
    if (!completedLessons.includes(lessonNum)) {
      completedLessons.push(lessonNum);
      localStorage.setItem('ttr_completed_lessons', JSON.stringify(completedLessons));
    }
  }

  // Add Copy Button to Code Pre Blocks
  document.querySelectorAll('pre').forEach(pre => {
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copia';
    copyBtn.className = 'btn-copy-code';
    copyBtn.style.position = 'absolute';
    copyBtn.style.top = '8px';
    copyBtn.style.right = '8px';
    copyBtn.style.fontSize = '0.75rem';
    copyBtn.style.padding = '3px 8px';
    copyBtn.style.background = 'var(--bg-tertiary)';
    copyBtn.style.color = 'var(--text-secondary)';
    copyBtn.style.border = '1px solid var(--border-color)';
    copyBtn.style.borderRadius = '4px';
    copyBtn.style.cursor = 'pointer';

    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(pre.innerText).then(() => {
        copyBtn.textContent = 'Copiato!';
        copyBtn.style.color = 'var(--accent-green)';
        setTimeout(() => {
          copyBtn.textContent = 'Copia';
          copyBtn.style.color = 'var(--text-secondary)';
        }, 2000);
      });
    });

    wrapper.appendChild(copyBtn);
  });
});
