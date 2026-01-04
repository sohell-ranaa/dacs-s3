/**
 * DACS Thesis Document Viewer
 * Main application controller
 */

const ThesisApp = {
    // Configuration
    config: {
        contentPath: 'content/',
        imagePath: 'images/',
        wordsPerPage: 275,
        scrollOffset: 80,
        debounceDelay: 100
    },

    // State
    state: {
        metadata: null,
        chapters: [],
        references: [],
        annex: null,
        currentSection: null,
        isLoading: true,
        tocBuilt: false
    },

    // DOM Elements
    elements: {
        content: null,
        tocNav: null,
        sidebar: null,
        overlay: null,
        loading: null,
        wordCount: null,
        pageCount: null,
        pageIndicator: null
    },

    // Appendix configuration
    appendices: [
        { id: 'appendix-a', title: 'Appendix A: Ensemble Methods Notebook', file: 'notebooks/ensemble.html', student: 'Student 1 (TP000001)' },
        { id: 'appendix-b', title: 'Appendix B: Non-Linear Methods Notebook', file: 'notebooks/nonlinear.html', student: 'Student 2 (TP000002)' },
        { id: 'appendix-c', title: 'Appendix C: Support Vector Methods Notebook', file: 'notebooks/supportvector.html', student: 'Student 3 (TP000003)' }
    ],

    /**
     * Initialize the application
     */
    async init() {
        this.cacheElements();
        this.bindEvents();

        try {
            await this.loadContent();
            this.renderDocument();
            this.buildToc();
            this.updateStats();
            this.hideLoading();
            this.initScrollSpy();
            this.initFirstAppendix();
            this.initDocumentTocLinks();
            this.initLightbox();
            this.initBackToTop();
            this.initReadingProgress();
            this.highlightBestValues();
        } catch (error) {
            console.error('Failed to initialize thesis viewer:', error);
            this.showError('Failed to load document. Please refresh the page.');
        }
    },

    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements.content = document.getElementById('thesisContent');
        this.elements.tocNav = document.getElementById('tocNav');
        this.elements.sidebar = document.getElementById('sidebar');
        this.elements.overlay = document.getElementById('sidebarOverlay');
        this.elements.loading = document.getElementById('contentLoading');
        this.elements.wordCount = document.getElementById('wordCount');
        this.elements.pageCount = document.getElementById('pageCount');
        this.elements.pageIndicator = document.getElementById('pageIndicator');
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Scroll spy
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => this.updateActiveSection(), this.config.debounceDelay);
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeToc();
                this.closeLightbox();
            }
        });

        // Resize handler
        window.addEventListener('resize', () => {
            if (window.innerWidth > 767) {
                this.closeToc();
            }
        });
    },

    /**
     * Load all content from JSON files
     */
    async loadContent() {
        const files = [
            'metadata.json',
            'chapter1.json',
            'chapter2.json',
            'chapter3.json',
            'chapter4.json',
            'chapter5.json',
            'references.json',
            'annex.json'
        ];

        const responses = await Promise.all(
            files.map(file =>
                fetch(this.config.contentPath + file)
                    .then(res => {
                        if (!res.ok) throw new Error(`Failed to load ${file}`);
                        return res.json();
                    })
                    .catch(err => {
                        console.warn(`Optional file ${file} not found`);
                        return null;
                    })
            )
        );

        this.state.metadata = responses[0];
        this.state.chapters = responses.slice(1, 6);
        this.state.references = responses[6];
        this.state.annex = responses[7];
    },

    /**
     * Render the complete document
     */
    renderDocument() {
        let html = '';

        // Cover Page
        html += this.renderCoverPage();

        // Declaration of Originality
        html += this.renderDeclaration();

        // Acknowledgments
        html += this.renderAcknowledgments();

        // Table of Contents
        html += this.renderTableOfContents();

        // List of Figures
        html += this.renderListOfFigures();

        // List of Tables
        html += this.renderListOfTables();

        // List of Abbreviations
        html += this.renderAbbreviations();

        // Abstract
        html += this.renderAbstract();

        // Chapters
        this.state.chapters.forEach(chapter => {
            html += this.renderChapter(chapter);
        });

        // References
        html += this.renderReferences();

        // Appendices
        html += this.renderAppendices();

        this.elements.content.innerHTML = html;
    },

    /**
     * Render cover page
     */
    renderCoverPage() {
        const meta = this.state.metadata;
        return `
            <section class="cover-page" id="cover">
                <div class="cover-institution">${meta.institution}</div>
                <h1 class="cover-title">${meta.title}</h1>
                <p class="cover-subtitle">${meta.subtitle}</p>

                <div class="cover-authors">
                    ${meta.authors.map(author => `
                        <div class="author-card">
                            <div class="author-name">${author.name}</div>
                            <div class="author-id">${author.id}</div>
                            <div class="author-role">${author.role}</div>
                        </div>
                    `).join('')}
                </div>

                <div class="cover-meta">
                    <span><strong>Supervisor:</strong> ${meta.supervisor}</span>
                    <span><strong>Module:</strong> ${meta.module}</span>
                    <span><strong>Date:</strong> ${meta.date}</span>
                </div>
            </section>
        `;
    },

    /**
     * Render declaration of originality
     */
    renderDeclaration() {
        const meta = this.state.metadata;
        if (!meta.declaration) return '';

        return `
            <section class="thesis-section declaration-section" id="declaration">
                <h2 class="chapter-title">Declaration of Originality</h2>
                <div class="declaration-content">
                    <p>${meta.declaration}</p>
                    <div class="signature-grid">
                        ${meta.authors.map(author => `
                            <div class="signature-block">
                                <div class="signature-line"></div>
                                <div class="signature-name">${author.name}</div>
                                <div class="signature-id">${author.id}</div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="declaration-date">
                        <strong>Date:</strong> ${meta.date}
                    </div>
                </div>
            </section>
        `;
    },

    /**
     * Render acknowledgments
     */
    renderAcknowledgments() {
        const meta = this.state.metadata;
        if (!meta.acknowledgments) return '';

        return `
            <section class="thesis-section acknowledgments-section" id="acknowledgments">
                <h2 class="chapter-title">Acknowledgments</h2>
                <div class="acknowledgments-content">
                    <p>${meta.acknowledgments}</p>
                </div>
            </section>
        `;
    },

    /**
     * Render table of contents
     */
    renderTableOfContents() {
        let html = `
            <section class="thesis-section toc-section" id="toc">
                <h2 class="chapter-title">Table of Contents</h2>
                <nav class="toc-content">
        `;

        this.state.chapters.forEach(chapter => {
            html += `
                <a href="#${chapter.chapter_key}" class="toc-chapter-link">
                    <span>Chapter ${chapter.chapter_number}: ${chapter.title}</span>
                    <span class="toc-page"></span>
                </a>
            `;

            if (chapter.sections) {
                chapter.sections.forEach(section => {
                    html += `
                        <a href="#section-${section.section_key}" class="toc-section-link">
                            <span>${section.section_key} ${section.title}</span>
                            <span class="toc-page"></span>
                        </a>
                    `;
                });
            }
        });

        html += `
                    <a href="#references" class="toc-chapter-link">
                        <span>References</span>
                        <span class="toc-page"></span>
                    </a>
                    <a href="#appendices" class="toc-chapter-link">
                        <span>Appendices</span>
                        <span class="toc-page"></span>
                    </a>
                    ${this.appendices.map(app => `
                        <a href="#${app.id}" class="toc-section-link">
                            <span>${app.title}</span>
                            <span class="toc-page"></span>
                        </a>
                    `).join('')}
                </nav>
            </section>
        `;

        return html;
    },

    /**
     * Render list of figures
     */
    renderListOfFigures() {
        const figures = [];

        this.state.chapters.forEach(chapter => {
            if (chapter.sections) {
                chapter.sections.forEach(section => {
                    if (section.figures) {
                        section.figures.forEach(fig => {
                            figures.push({
                                id: fig.id,
                                caption: fig.caption,
                                chapter: chapter.chapter_number
                            });
                        });
                    }
                });
            }
        });

        if (figures.length === 0) return '';

        return `
            <section class="thesis-section" id="list-of-figures">
                <h2 class="chapter-title">List of Figures</h2>
                <div class="list-content">
                    ${figures.map((fig, idx) => `
                        <a href="#${fig.id}" class="list-item">
                            <span class="list-item-text">Figure ${idx + 1}: ${fig.caption}</span>
                            <span class="list-item-dots"></span>
                            <span class="list-item-page"></span>
                        </a>
                    `).join('')}
                </div>
            </section>
        `;
    },

    /**
     * Render list of tables
     */
    renderListOfTables() {
        const tables = [];

        this.state.chapters.forEach(chapter => {
            if (chapter.sections) {
                chapter.sections.forEach(section => {
                    if (section.tables) {
                        section.tables.forEach(table => {
                            tables.push({
                                id: table.id,
                                caption: table.caption,
                                chapter: chapter.chapter_number
                            });
                        });
                    }
                });
            }
        });

        if (tables.length === 0) return '';

        return `
            <section class="thesis-section" id="list-of-tables">
                <h2 class="chapter-title">List of Tables</h2>
                <div class="list-content">
                    ${tables.map((table, idx) => `
                        <a href="#${table.id}" class="list-item">
                            <span class="list-item-text">Table ${idx + 1}: ${table.caption}</span>
                            <span class="list-item-dots"></span>
                            <span class="list-item-page"></span>
                        </a>
                    `).join('')}
                </div>
            </section>
        `;
    },

    /**
     * Render list of abbreviations
     */
    renderAbbreviations() {
        const meta = this.state.metadata;
        if (!meta.abbreviations || meta.abbreviations.length === 0) return '';

        return `
            <section class="thesis-section abbreviations-section" id="abbreviations">
                <h2 class="chapter-title">List of Abbreviations</h2>
                <div class="abbreviations-content">
                    <dl class="abbreviations-list">
                        ${meta.abbreviations.map(item => `
                            <div class="abbr-item">
                                <dt>${item.abbr}</dt>
                                <dd>${item.full}</dd>
                            </div>
                        `).join('')}
                    </dl>
                </div>
            </section>
        `;
    },

    /**
     * Render abstract
     */
    renderAbstract() {
        const meta = this.state.metadata;
        if (!meta.abstract) return '';

        return `
            <section class="thesis-section" id="abstract">
                <div class="abstract-box">
                    <h3>Abstract</h3>
                    <p>${meta.abstract}</p>
                    ${meta.keywords ? `
                        <div class="abstract-keywords">
                            <strong>Keywords:</strong> ${meta.keywords.join(', ')}
                        </div>
                    ` : ''}
                </div>
            </section>
        `;
    },

    /**
     * Render a chapter
     */
    renderChapter(chapter) {
        let html = `
            <section class="thesis-section chapter" id="${chapter.chapter_key}">
                <h2 class="chapter-title">
                    <span class="chapter-number">Chapter ${chapter.chapter_number}</span>
                    ${chapter.title}
                </h2>
        `;

        if (chapter.introduction) {
            html += `<div class="chapter-intro">${chapter.introduction}</div>`;
        }

        if (chapter.sections) {
            chapter.sections.forEach(section => {
                html += this.renderSection(section);
            });
        }

        html += '</section>';
        return html;
    },

    /**
     * Render a section
     */
    renderSection(section) {
        let html = `
            <div class="section" id="section-${section.section_key}">
                <h3 class="section-title">
                    <span class="section-number">${section.section_key}</span>
                    ${section.title}
                </h3>
                <div class="section-content">
                    ${section.content_html || ''}
                </div>
        `;

        // Render figures
        if (section.figures) {
            section.figures.forEach(fig => {
                html += this.renderFigure(fig);
            });
        }

        // Render tables
        if (section.tables) {
            section.tables.forEach(table => {
                html += this.renderTable(table);
            });
        }

        // Render subsections
        if (section.subsections) {
            section.subsections.forEach(subsection => {
                html += this.renderSubsection(subsection);
            });
        }

        html += '</div>';
        return html;
    },

    /**
     * Render a subsection
     */
    renderSubsection(subsection) {
        let html = `
            <div class="subsection" id="section-${subsection.section_key}">
                <h4 class="subsection-title">
                    <span class="section-number">${subsection.section_key}</span>
                    ${subsection.title}
                </h4>
                <div class="subsection-content">
                    ${subsection.content_html || ''}
                </div>
        `;

        // Render figures
        if (subsection.figures) {
            subsection.figures.forEach(fig => {
                html += this.renderFigure(fig);
            });
        }

        // Render tables
        if (subsection.tables) {
            subsection.tables.forEach(table => {
                html += this.renderTable(table);
            });
        }

        html += '</div>';
        return html;
    },

    /**
     * Render a figure (APA style: caption at bottom, followed by description)
     */
    renderFigure(fig) {
        const imagePath = fig.src.startsWith('http') ? fig.src : this.config.imagePath + fig.src;
        return `
            <figure class="thesis-figure" id="${fig.id}">
                <img src="${imagePath}" alt="${fig.caption}" loading="lazy">
                <figcaption class="figure-caption">
                    <span class="figure-label">Figure ${fig.number || ''}</span>
                    <span class="figure-title">${fig.caption}</span>
                </figcaption>
                ${fig.description ? `<p class="figure-description">${fig.description}</p>` : ''}
            </figure>
        `;
    },

    /**
     * Render a table (APA style: caption at bottom, followed by description)
     */
    renderTable(table) {
        let html = `
            <div class="thesis-table-wrapper" id="${table.id}">
                <table class="thesis-table">
        `;

        if (table.headers) {
            html += '<thead><tr>';
            table.headers.forEach(header => {
                html += `<th>${header}</th>`;
            });
            html += '</tr></thead>';
        }

        if (table.rows) {
            html += '<tbody>';
            table.rows.forEach(row => {
                html += '<tr>';
                row.forEach((cell, idx) => {
                    const isFirst = idx === 0;
                    html += isFirst ? `<td class="row-header">${cell}</td>` : `<td>${cell}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody>';
        }

        html += `</table>
                <div class="table-caption">
                    <span class="table-label">Table ${table.number || ''}</span>
                    <span class="table-title">${table.caption}</span>
                </div>
                ${table.description ? `<p class="table-description">${table.description}</p>` : ''}
            </div>`;
        return html;
    },

    /**
     * Render references
     */
    renderReferences() {
        const refs = this.state.references;
        if (!refs || !refs.references) return '';

        return `
            <section class="thesis-section references-section" id="references">
                <h2 class="chapter-title">References</h2>
                <div class="references-list">
                    ${refs.references.map((ref, idx) => `
                        <div class="reference-item" id="ref-${idx + 1}">
                            <span class="reference-number">${idx + 1}</span>
                            <div class="reference-text">
                                ${ref.citation}
                                ${ref.url ? `<a href="${ref.url}" target="_blank" rel="noopener" class="reference-link">[Link]</a>` : ''}
                                ${ref.doi ? `<a href="https://doi.org/${ref.doi}" target="_blank" rel="noopener" class="reference-link">[DOI]</a>` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </section>
        `;
    },

    /**
     * Render appendices section as inline pages (no tabs)
     */
    renderAppendices() {
        return `
            <section class="thesis-section appendices-section" id="appendices">
                <h2 class="chapter-title">
                    <span class="chapter-number">Appendices</span>
                    Individual Notebook Contributions
                </h2>
                <p class="appendix-intro">
                    The following appendices contain the complete Jupyter notebooks with executable code and outputs
                    for each team member's individual contribution. These notebooks demonstrate the full implementation
                    process including data preprocessing, model training, hyperparameter tuning, and evaluation.
                </p>

                <!-- Inline Appendix Sections -->
                ${this.appendices.map((app, idx) => `
                    <div class="appendix-page" id="${app.id}" data-loaded="false">
                        <div class="appendix-header">
                            <h3 class="appendix-title">${app.title}</h3>
                            <div class="appendix-meta">
                                <span class="appendix-author">${app.student}</span>
                                <a href="${app.file}" target="_blank" class="appendix-open-btn">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                                        <polyline points="15 3 21 3 21 9"/>
                                        <line x1="10" y1="14" x2="21" y2="3"/>
                                    </svg>
                                    Open in New Tab
                                </a>
                            </div>
                        </div>
                        <div class="appendix-iframe-container">
                            <div class="appendix-loading">
                                <div class="loading-spinner"></div>
                                <p>Loading notebook...</p>
                            </div>
                            <iframe class="appendix-iframe"
                                    data-src="${app.file}"
                                    title="${app.title}"
                                    sandbox="allow-scripts allow-same-origin"></iframe>
                        </div>
                    </div>
                `).join('')}
            </section>
        `;
    },

    /**
     * Load a specific appendix iframe
     */
    loadAppendix(appendixId) {
        const page = document.getElementById(appendixId);
        if (!page || page.dataset.loaded === 'true') return;

        const iframe = page.querySelector('.appendix-iframe');
        const loading = page.querySelector('.appendix-loading');

        if (iframe && iframe.dataset.src) {
            iframe.src = iframe.dataset.src;
            iframe.onload = () => {
                loading.style.display = 'none';
                iframe.style.opacity = '1';
                page.dataset.loaded = 'true';
            };
        }
    },

    /**
     * Load all appendices (for inline display)
     */
    loadAllAppendices() {
        this.appendices.forEach(app => {
            this.loadAppendix(app.id);
        });
    },

    /**
     * Build sidebar table of contents
     */
    buildToc() {
        let html = '';

        // Front matter
        html += `
            <div class="toc-group">
                <a href="#cover" class="toc-chapter-link" data-section="cover">Cover</a>
                <a href="#declaration" class="toc-section-link" data-section="declaration">Declaration</a>
                <a href="#acknowledgments" class="toc-section-link" data-section="acknowledgments">Acknowledgments</a>
                <a href="#toc" class="toc-section-link" data-section="toc">Table of Contents</a>
                <a href="#abbreviations" class="toc-section-link" data-section="abbreviations">Abbreviations</a>
                <a href="#abstract" class="toc-section-link" data-section="abstract">Abstract</a>
            </div>
        `;

        // Chapters
        this.state.chapters.forEach(chapter => {
            html += `
                <div class="toc-group">
                    <a href="#${chapter.chapter_key}" class="toc-chapter-link" data-section="${chapter.chapter_key}">
                        ${chapter.chapter_number}. ${chapter.title}
                    </a>
            `;

            if (chapter.sections) {
                chapter.sections.forEach(section => {
                    html += `
                        <a href="#section-${section.section_key}" class="toc-section-link" data-section="section-${section.section_key}">
                            ${section.section_key} ${section.title}
                        </a>
                    `;
                });
            }

            html += '</div>';
        });

        // References
        html += `
            <div class="toc-group">
                <a href="#references" class="toc-chapter-link" data-section="references">References</a>
            </div>
        `;

        // Appendices
        html += `
            <div class="toc-group">
                <a href="#appendices" class="toc-chapter-link" data-section="appendices">Appendices</a>
                ${this.appendices.map(app => `
                    <a href="#${app.id}" class="toc-section-link" data-section="${app.id}">
                        ${app.title.replace('Appendix ', '')}
                    </a>
                `).join('')}
            </div>
        `;

        this.elements.tocNav.innerHTML = html;
        this.state.tocBuilt = true;

        // Disable click handlers - make TOC non-clickable
        this.elements.tocNav.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                // Do nothing - navigation disabled
            });
            link.style.cursor = 'default';
            link.style.pointerEvents = 'none';
        });
    },

    /**
     * Scroll to section
     */
    scrollToSection(id) {
        // Handle appendix sections - load if needed, then scroll directly to the appendix
        if (id.startsWith('appendix-')) {
            this.loadAppendix(id);
        }

        const element = document.getElementById(id);
        if (element) {
            const top = element.offsetTop - this.config.scrollOffset;
            window.scrollTo({
                top: top,
                behavior: 'smooth'
            });
        }
    },

    /**
     * Initialize scroll spy
     */
    initScrollSpy() {
        this.updateActiveSection();
    },

    /**
     * Update active section based on scroll position
     */
    updateActiveSection() {
        const sections = this.elements.content.querySelectorAll('[id]');
        const scrollPos = window.scrollY + this.config.scrollOffset + 50;

        let currentSection = null;

        sections.forEach(section => {
            if (section.offsetTop <= scrollPos) {
                currentSection = section.id;
            }
        });

        if (currentSection !== this.state.currentSection) {
            this.state.currentSection = currentSection;
            this.highlightTocItem(currentSection);
        }
    },

    /**
     * Highlight active TOC item
     */
    highlightTocItem(sectionId) {
        if (!this.state.tocBuilt) return;

        // Remove all active classes
        this.elements.tocNav.querySelectorAll('.active').forEach(el => {
            el.classList.remove('active');
        });

        // Add active class to current
        const activeLink = this.elements.tocNav.querySelector(`[data-section="${sectionId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');

            // Scroll TOC to show active item
            const tocNav = this.elements.tocNav;
            const linkTop = activeLink.offsetTop;
            const tocHeight = tocNav.clientHeight;
            const scrollTop = tocNav.scrollTop;

            if (linkTop < scrollTop || linkTop > scrollTop + tocHeight - 50) {
                tocNav.scrollTo({
                    top: linkTop - tocHeight / 2,
                    behavior: 'smooth'
                });
            }
        }
    },

    /**
     * Calculate and update document stats
     */
    updateStats() {
        const text = this.elements.content.textContent || '';
        const words = text.trim().split(/\s+/).filter(word => word.length > 0).length;
        const pages = Math.ceil(words / this.config.wordsPerPage);

        this.elements.wordCount.textContent = words.toLocaleString();
        this.elements.pageCount.textContent = pages;

        // Store total pages for page indicator
        this.state.totalPages = pages;

        // Update page indicator
        if (this.elements.pageIndicator) {
            this.elements.pageIndicator.querySelector('.total-pages').textContent = pages;
        }

        // Initialize page scroll tracking
        this.initPageTracking();
    },

    /**
     * Initialize page number tracking on scroll
     */
    initPageTracking() {
        const updatePageNumber = () => {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = docHeight > 0 ? scrollTop / docHeight : 0;
            const currentPage = Math.max(1, Math.ceil(scrollPercent * this.state.totalPages));

            if (this.elements.pageIndicator) {
                this.elements.pageIndicator.querySelector('.current-page').textContent =
                    Math.min(currentPage, this.state.totalPages);
            }
        };

        window.addEventListener('scroll', updatePageNumber);
        updatePageNumber(); // Initial update
    },

    /**
     * Initialize appendix lazy loading (load all when section is visible)
     */
    initFirstAppendix() {
        // Load all appendices when user scrolls near the appendix section
        const appendixSection = document.getElementById('appendices');
        if (!appendixSection) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.loadAllAppendices();
                    observer.disconnect();
                }
            });
        }, { rootMargin: '200px' });

        observer.observe(appendixSection);
    },

    /**
     * Initialize click handlers for in-document TOC links - disabled
     */
    initDocumentTocLinks() {
        // Disable click handlers for TOC, List of Figures, and List of Tables
        const sections = ['toc', 'list-of-figures', 'list-of-tables'];

        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (!section) return;

            section.querySelectorAll('a[href^="#"]').forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    // Do nothing - navigation disabled
                });
                link.style.cursor = 'default';
                link.style.pointerEvents = 'none';
            });
        });
    },

    /**
     * Toggle sidebar (mobile)
     */
    toggleToc() {
        this.elements.sidebar.classList.toggle('active');
        this.elements.overlay.classList.toggle('active');
        document.body.style.overflow = this.elements.sidebar.classList.contains('active') ? 'hidden' : '';
    },

    /**
     * Close sidebar
     */
    closeToc() {
        this.elements.sidebar.classList.remove('active');
        this.elements.overlay.classList.remove('active');
        document.body.style.overflow = '';
    },

    /**
     * Hide loading state
     */
    hideLoading() {
        this.elements.loading.style.display = 'none';
        this.state.isLoading = false;
    },

    /**
     * Show error message
     */
    showError(message) {
        this.elements.content.innerHTML = `
            <div class="error-state">
                <div class="error-icon">!</div>
                <h3>Error Loading Document</h3>
                <p>${message}</p>
                <button onclick="location.reload()" class="btn-retry">Retry</button>
            </div>
        `;
        this.hideLoading();
    },

    /**
     * Get all content for export
     */
    getExportData() {
        return {
            metadata: this.state.metadata,
            chapters: this.state.chapters,
            references: this.state.references,
            annex: this.state.annex
        };
    },

    /**
     * Initialize lightbox for figures
     */
    initLightbox() {
        // Create lightbox overlay if not exists
        if (!document.getElementById('lightboxOverlay')) {
            const overlay = document.createElement('div');
            overlay.id = 'lightboxOverlay';
            overlay.className = 'lightbox-overlay';
            overlay.innerHTML = `
                <button class="lightbox-close" onclick="ThesisApp.closeLightbox()">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
                <div class="lightbox-content">
                    <img id="lightboxImage" src="" alt="">
                    <div class="lightbox-caption" id="lightboxCaption"></div>
                </div>
            `;
            document.body.appendChild(overlay);

            // Close on overlay click
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.closeLightbox();
                }
            });
        }

        // Add click handlers to figures
        document.querySelectorAll('.thesis-figure').forEach(figure => {
            figure.addEventListener('click', () => {
                const img = figure.querySelector('img');
                const caption = figure.querySelector('.figure-caption');
                if (img) {
                    this.openLightbox(img.src, caption ? caption.textContent : '');
                }
            });
        });
    },

    /**
     * Open lightbox with image
     */
    openLightbox(src, caption) {
        const overlay = document.getElementById('lightboxOverlay');
        const img = document.getElementById('lightboxImage');
        const captionEl = document.getElementById('lightboxCaption');

        if (overlay && img) {
            img.src = src;
            captionEl.textContent = caption;
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    },

    /**
     * Close lightbox
     */
    closeLightbox() {
        const overlay = document.getElementById('lightboxOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    },

    /**
     * Initialize back to top button
     */
    initBackToTop() {
        // Create button if not exists
        if (!document.getElementById('backToTop')) {
            const btn = document.createElement('button');
            btn.id = 'backToTop';
            btn.className = 'back-to-top';
            btn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="18 15 12 9 6 15"/>
                </svg>
            `;
            btn.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
            document.body.appendChild(btn);
        }

        // Show/hide on scroll
        const btn = document.getElementById('backToTop');
        window.addEventListener('scroll', () => {
            if (window.scrollY > 500) {
                btn.classList.add('visible');
            } else {
                btn.classList.remove('visible');
            }
        });
    },

    /**
     * Initialize reading progress bar
     */
    initReadingProgress() {
        // Create progress bar if not exists
        if (!document.getElementById('readingProgress')) {
            const progress = document.createElement('div');
            progress.id = 'readingProgress';
            progress.className = 'reading-progress';
            progress.innerHTML = '<div class="reading-progress-bar"></div>';
            document.body.appendChild(progress);
        }

        const bar = document.querySelector('.reading-progress-bar');
        window.addEventListener('scroll', () => {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
            bar.style.width = `${progress}%`;
        });
    },

    /**
     * Highlight best values in tables
     */
    highlightBestValues() {
        document.querySelectorAll('.thesis-table').forEach(table => {
            const rows = table.querySelectorAll('tbody tr');
            if (rows.length === 0) return;

            // Get number of columns (excluding first row header column)
            const firstRow = rows[0];
            const numCols = firstRow.querySelectorAll('td').length;

            // For each numeric column, find the best value
            for (let col = 1; col < numCols; col++) {
                let bestVal = -Infinity;
                let bestCell = null;

                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells[col]) {
                        const val = parseFloat(cells[col].textContent);
                        if (!isNaN(val) && val > bestVal) {
                            bestVal = val;
                            bestCell = cells[col];
                        }
                    }
                });

                // Highlight best cell (for positive metrics like accuracy)
                if (bestCell && bestVal > 0) {
                    bestCell.classList.add('best-value');
                    bestCell.closest('tr').classList.add('winner-row');
                }
            }
        });
    }
};

// Make globally available
window.ThesisApp = ThesisApp;
