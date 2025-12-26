// DACS Machine Learning Study Guide - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize accordions
    initAccordions();

    // Initialize quiz functionality
    initQuiz();

    // Initialize progress tracking
    initProgress();

    // Smooth scrolling
    initSmoothScroll();

    // Active nav highlighting
    initActiveNav();
});

// Accordion functionality
function initAccordions() {
    const accordions = document.querySelectorAll('.accordion');
    accordions.forEach(accordion => {
        const header = accordion.querySelector('.accordion-header');
        header.addEventListener('click', () => {
            accordion.classList.toggle('active');
        });
    });
}

// Quiz functionality
function initQuiz() {
    const quizForms = document.querySelectorAll('.quiz-form');
    quizForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            checkAnswers(this);
        });
    });
}

function checkAnswers(form) {
    const questions = form.querySelectorAll('.quiz-question');
    let correct = 0;
    let total = questions.length;

    questions.forEach(q => {
        const selected = q.querySelector('input:checked');
        const correctAnswer = q.dataset.answer;

        // Reset previous feedback
        q.querySelectorAll('label').forEach(l => {
            l.classList.remove('correct', 'incorrect');
        });

        if (selected) {
            if (selected.value === correctAnswer) {
                selected.parentElement.classList.add('correct');
                correct++;
            } else {
                selected.parentElement.classList.add('incorrect');
                // Show correct answer
                q.querySelector(`input[value="${correctAnswer}"]`)
                    .parentElement.classList.add('correct');
            }
        }
    });

    // Show result
    const result = form.querySelector('.quiz-result') || document.createElement('div');
    result.className = 'quiz-result';
    result.innerHTML = `
        <div class="key-concept">
            <h4>Quiz Result</h4>
            <p>You got <strong>${correct}</strong> out of <strong>${total}</strong> correct (${Math.round(correct/total*100)}%)</p>
        </div>
    `;
    if (!form.querySelector('.quiz-result')) {
        form.appendChild(result);
    }
}

// Progress tracking
function initProgress() {
    const sections = document.querySelectorAll('.section');
    const progressBar = document.querySelector('.progress');

    if (!progressBar) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const index = Array.from(sections).indexOf(entry.target);
                const progress = ((index + 1) / sections.length) * 100;
                progressBar.style.width = `${progress}%`;
            }
        });
    }, { threshold: 0.5 });

    sections.forEach(section => observer.observe(section));
}

// Smooth scrolling
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Active navigation highlighting
function initActiveNav() {
    const sections = document.querySelectorAll('.section[id]');
    const navLinks = document.querySelectorAll('.sidebar-nav a');

    if (sections.length === 0 || navLinks.length === 0) return;

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (scrollY >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// Toggle dark mode
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

// Load dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}

// Copy code blocks
function copyCode(button) {
    const code = button.parentElement.querySelector('code');
    navigator.clipboard.writeText(code.textContent);
    button.textContent = 'Copied!';
    setTimeout(() => button.textContent = 'Copy', 2000);
}

// Show/Hide answers
function toggleAnswer(id) {
    const answer = document.getElementById(id);
    answer.style.display = answer.style.display === 'none' ? 'block' : 'none';
}
