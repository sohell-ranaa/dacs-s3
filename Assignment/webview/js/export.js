/**
 * DACS Thesis Document - APA 7th Edition Academic DOCX Export
 * Professional academic styling with proper notebook embedding
 */

const ThesisExport = {
    docx: null,
    figureNumber: 0,
    tableNumber: 0,
    shortTitle: 'MACHINE LEARNING FOR INTRUSION DETECTION',  // Running header

    // Click protection - requires 6 consecutive clicks
    clickCount: 0,
    lastClickTime: 0,
    clickTimeout: null,
    requiredClicks: 6,
    clickWindow: 2000,  // 2 seconds window for consecutive clicks

    // APA Style - Clean academic colors
    colors: {
        black: '000000',
        darkGray: '333333',
        gray: '666666',
        lightGray: '999999',
        tableBorder: '000000',
        tableHeaderBg: 'F2F2F2',
        tableAltRow: 'FAFAFA',
        link: '0000FF'
    },

    // Font sizes (in half-points) - Optimized for space
    sizes: {
        title: 28,          // 14pt for title
        heading1: 24,       // 12pt
        heading2: 24,       // 12pt
        heading3: 24,       // 12pt
        body: 22,           // 11pt body text (saves space)
        table: 20,          // 10pt for tables (compact)
        caption: 20,        // 10pt for captions
        small: 18,          // 9pt for notes
        code: 18,           // 9pt for code
        header: 20          // 10pt for header/footer
    },

    // Line spacing - 1.5 instead of double saves ~25% space
    lineSpacing: {
        body: 360,          // 1.5 line spacing
        heading: 276,       // 1.15 line spacing for headings
        table: 240          // Single spacing for tables
    },

    /**
     * Export document to DOCX - requires 6 consecutive clicks
     */
    async exportToDocx() {
        const now = Date.now();
        const btn = document.getElementById('exportBtn');

        // Check if click is within time window
        if (now - this.lastClickTime > this.clickWindow) {
            // Reset counter if too much time passed
            this.clickCount = 0;
        }

        this.lastClickTime = now;
        this.clickCount++;

        // Clear any existing timeout
        if (this.clickTimeout) {
            clearTimeout(this.clickTimeout);
        }

        // Check if we have enough clicks (secret - no visual feedback)
        const remaining = this.requiredClicks - this.clickCount;
        if (remaining > 0) {
            // Reset counter after timeout (silently)
            this.clickTimeout = setTimeout(() => {
                this.clickCount = 0;
            }, this.clickWindow);
            return;
        }

        // Reset click counter
        this.clickCount = 0;

        // Proceed with actual export
        if (typeof docx === 'undefined') {
            alert('Export library not available. Please refresh the page.');
            return;
        }

        this.docx = docx;
        this.figureNumber = 0;
        this.tableNumber = 0;

        const originalContent = btn.innerHTML;

        try {
            btn.innerHTML = '<span>Exporting...</span>';
            btn.disabled = true;

            const data = ThesisApp.getExportData();
            const doc = await this.buildDocument(data);
            const blob = await this.docx.Packer.toBlob(doc);

            const filename = `DACS_Assignment_${data.metadata.date.replace(/\s/g, '_')}.docx`;
            saveAs(blob, filename);

            btn.innerHTML = '<span>Downloaded!</span>';
            setTimeout(() => {
                btn.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    <span>Export DOCX</span>
                `;
                btn.disabled = false;
            }, 2000);

        } catch (error) {
            console.error('Export failed:', error);
            btn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                <span>Export DOCX</span>
            `;
            btn.disabled = false;
            alert('Export failed: ' + error.message);
        }
    },

    /**
     * Build the complete document with multiple sections
     * APA 7 Student Paper Format: Continuous page numbering throughout
     */
    async buildDocument(data) {
        const sections = [];
        const isStudentPaper = true;  // Set to false for professional paper format

        // Section 1: Title page (with page number for student papers)
        const titleContent = this.buildTitlePage(data.metadata);
        sections.push({
            properties: this.getSectionProperties({ titlePage: true }),
            headers: this.createHeaders(this.shortTitle, isStudentPaper),
            children: titleContent
        });

        // Section 2: Front matter (continuous page numbering for student papers)
        const frontMatter = [];
        frontMatter.push(...this.buildDeclaration(data.metadata));
        frontMatter.push(...this.buildAcknowledgments(data.metadata));
        frontMatter.push(...this.buildTableOfContents());
        frontMatter.push(...this.buildListOfFigures(data.chapters));
        frontMatter.push(...this.buildListOfTables(data.chapters));
        frontMatter.push(...this.buildAbstract(data.metadata));

        sections.push({
            properties: this.getSectionProperties({
                pageNumberFormat: 'decimal',
                hasHeader: true
            }),
            headers: this.createHeaders(this.shortTitle, isStudentPaper),
            children: frontMatter
        });

        // Section 3: Main content (continuous page numbering)
        const mainContent = [];
        for (const chapter of data.chapters) {
            mainContent.push(...await this.buildChapter(chapter));
        }
        if (data.references?.references) {
            mainContent.push(...this.buildReferences(data.references.references));
        }
        // Appendices excluded - to be added separately
        // mainContent.push(...await this.buildAppendices());

        sections.push({
            properties: this.getSectionProperties({
                pageNumberFormat: 'decimal',
                hasHeader: true
            }),
            headers: this.createHeaders(this.shortTitle, isStudentPaper),
            children: mainContent
        });

        return new this.docx.Document({
            styles: this.getAPAStyles(),
            numbering: this.getNumberingConfig(),
            sections: sections
        });
    },

    /**
     * Get section properties
     */
    getSectionProperties(options = {}) {
        const props = {
            page: {
                margin: {
                    top: 1152,    // 0.8 inch (saves space)
                    right: 1152,  // 0.8 inch
                    bottom: 1152, // 0.8 inch
                    left: 1152    // 0.8 inch
                }
            }
        };

        if (options.titlePage) {
            props.titlePage = true;
        }

        return props;
    },

    /**
     * Create headers for a section
     * APA 7 Student Paper: Page number only in upper right
     * APA 7 Professional Paper: Running head left, page number right
     */
    createHeaders(text, isStudentPaper = true) {
        if (isStudentPaper) {
            // Student paper: page number only, flush right
            return {
                default: new this.docx.Header({
                    children: [
                        new this.docx.Paragraph({
                            alignment: this.docx.AlignmentType.RIGHT,
                            spacing: { after: 0 },
                            children: [
                                new this.docx.TextRun({
                                    children: [this.docx.PageNumber.CURRENT],
                                    size: this.sizes.header,
                                    font: 'Times New Roman'
                                })
                            ]
                        })
                    ]
                })
            };
        } else {
            // Professional paper: running head left, page number right
            return {
                default: new this.docx.Header({
                    children: [
                        new this.docx.Paragraph({
                            alignment: this.docx.AlignmentType.LEFT,
                            spacing: { after: 0 },
                            tabStops: [{
                                type: this.docx.TabStopType.RIGHT,
                                position: 9360  // Right margin position (6.5" from left in twips)
                            }],
                            children: [
                                new this.docx.TextRun({
                                    text: text,
                                    size: this.sizes.header,
                                    font: 'Times New Roman'
                                }),
                                new this.docx.TextRun({
                                    text: '\t'  // Tab to right
                                }),
                                new this.docx.TextRun({
                                    children: [this.docx.PageNumber.CURRENT],
                                    size: this.sizes.header,
                                    font: 'Times New Roman'
                                })
                            ]
                        })
                    ]
                })
            };
        }
    },

    /**
     * Create footers with page numbers
     */
    createFooters(style = 'arabic') {
        return {
            default: new this.docx.Footer({
                children: [
                    new this.docx.Paragraph({
                        alignment: this.docx.AlignmentType.CENTER,
                        children: [
                            new this.docx.TextRun({
                                children: [
                                    style === 'roman'
                                        ? this.docx.PageNumber.CURRENT
                                        : this.docx.PageNumber.CURRENT
                                ],
                                size: this.sizes.header,
                                font: 'Times New Roman'
                            })
                        ]
                    })
                ]
            })
        };
    },

    /**
     * APA 7th Edition Document Styles
     */
    getAPAStyles() {
        return {
            default: {
                document: {
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.body,
                        color: this.colors.black
                    },
                    paragraph: {
                        spacing: { line: 340, after: 0 },  // 1.4 line spacing (compact)
                        alignment: this.docx.AlignmentType.LEFT
                    }
                }
            },
            paragraphStyles: [
                {
                    id: 'Title',
                    name: 'Title',
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.title,
                        bold: true
                    },
                    paragraph: {
                        alignment: this.docx.AlignmentType.CENTER,
                        spacing: { after: 120, line: 340 }
                    }
                },
                {
                    // APA Level 1: Centered, Bold
                    id: 'Heading1',
                    name: 'Heading 1',
                    basedOn: 'Normal',
                    next: 'Normal',
                    quickFormat: true,
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.heading1,
                        bold: true
                    },
                    paragraph: {
                        alignment: this.docx.AlignmentType.CENTER,
                        spacing: { before: 240, after: 120, line: 340 },
                        outlineLevel: 0
                    }
                },
                {
                    // APA Level 2: Left-aligned, Bold
                    id: 'Heading2',
                    name: 'Heading 2',
                    basedOn: 'Normal',
                    next: 'Normal',
                    quickFormat: true,
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.heading2,
                        bold: true
                    },
                    paragraph: {
                        alignment: this.docx.AlignmentType.LEFT,
                        spacing: { before: 180, after: 120, line: 340 },
                        outlineLevel: 1
                    }
                },
                {
                    // APA Level 3: Left-aligned, Bold, Italic
                    id: 'Heading3',
                    name: 'Heading 3',
                    basedOn: 'Normal',
                    next: 'Normal',
                    quickFormat: true,
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.heading3,
                        bold: true,
                        italics: true
                    },
                    paragraph: {
                        alignment: this.docx.AlignmentType.LEFT,
                        spacing: { before: 240, after: 120, line: 340 },
                        outlineLevel: 2
                    }
                },
                {
                    // APA Level 4: Indented, Bold, Title Case, Ending with Period
                    id: 'Heading4',
                    name: 'Heading 4',
                    basedOn: 'Normal',
                    next: 'Normal',
                    quickFormat: true,
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.body,
                        bold: true
                    },
                    paragraph: {
                        alignment: this.docx.AlignmentType.LEFT,
                        indent: { firstLine: 720 },
                        spacing: { before: 240, after: 0, line: 340 },
                        outlineLevel: 3
                    }
                },
                {
                    // APA Level 5: Indented, Bold, Italic, Title Case, Ending with Period
                    id: 'Heading5',
                    name: 'Heading 5',
                    basedOn: 'Normal',
                    next: 'Normal',
                    quickFormat: true,
                    run: {
                        font: 'Times New Roman',
                        size: this.sizes.body,
                        bold: true,
                        italics: true
                    },
                    paragraph: {
                        alignment: this.docx.AlignmentType.LEFT,
                        indent: { firstLine: 720 },
                        spacing: { before: 240, after: 0, line: 340 },
                        outlineLevel: 4
                    }
                }
            ]
        };
    },

    /**
     * Numbering configuration
     */
    getNumberingConfig() {
        return {
            config: [
                {
                    reference: 'numbered-list',
                    levels: [{
                        level: 0,
                        format: this.docx.LevelFormat.DECIMAL,
                        text: '%1.',
                        alignment: this.docx.AlignmentType.LEFT,
                        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                    }]
                },
                {
                    reference: 'bullet-list',
                    levels: [
                        {
                            level: 0,
                            format: this.docx.LevelFormat.BULLET,
                            text: '•',
                            alignment: this.docx.AlignmentType.LEFT,
                            style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                        },
                        {
                            level: 1,
                            format: this.docx.LevelFormat.BULLET,
                            text: '○',
                            alignment: this.docx.AlignmentType.LEFT,
                            style: { paragraph: { indent: { left: 1080, hanging: 360 } } }
                        }
                    ]
                }
            ]
        };
    },

    /**
     * APA Title Page
     */
    buildTitlePage(meta) {
        const elements = [];
        const P = this.docx.Paragraph;
        const T = this.docx.TextRun;
        const CENTER = this.docx.AlignmentType.CENTER;

        // Spacing from top
        for (let i = 0; i < 6; i++) {
            elements.push(new P({ spacing: { after: 120 } }));
        }

        // Title (Bold, Centered)
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 120, line: 340 },
            children: [new T({
                text: meta.title,
                bold: true,
                size: this.sizes.title,
                font: 'Times New Roman'
            })]
        }));

        // Subtitle
        if (meta.subtitle) {
            elements.push(new P({
                alignment: CENTER,
                spacing: { after: 480, line: 340 },
                children: [new T({
                    text: meta.subtitle,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
        }

        // Authors
        meta.authors.forEach(author => {
            elements.push(new P({
                alignment: CENTER,
                spacing: { after: 0, line: 340 },
                children: [new T({
                    text: author.name,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
        });

        elements.push(new P({ spacing: { after: 120 } }));

        // Institution
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 340 },
            children: [new T({
                text: meta.institution,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Module
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 340 },
            children: [new T({
                text: meta.module,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Supervisor
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 340 },
            children: [new T({
                text: `Supervisor: ${meta.supervisor}`,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Date
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 340 },
            children: [new T({
                text: meta.date,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // No page break needed - section break handles this
        return elements;
    },

    /**
     * Declaration Page
     */
    buildDeclaration(meta) {
        const elements = [];

        // Heading (APA Level 1 - Centered, Bold)
        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Declaration of Originality',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Declaration text
        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.LEFT,
            spacing: { after: 120, line: 340 },
            indent: { firstLine: 720 },
            children: [new this.docx.TextRun({
                text: meta.declaration,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Signatures
        elements.push(new this.docx.Paragraph({ spacing: { after: 480 } }));

        meta.authors.forEach(author => {
            elements.push(new this.docx.Paragraph({
                spacing: { after: 480, line: 340 },
                children: [new this.docx.TextRun({
                    text: '_'.repeat(50),
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
            elements.push(new this.docx.Paragraph({
                spacing: { after: 120, line: 340 },
                children: [new this.docx.TextRun({
                    text: `${author.name} (${author.id})`,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
        });

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * Acknowledgments Page
     */
    buildAcknowledgments(meta) {
        const elements = [];

        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Acknowledgments',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.LEFT,
            spacing: { after: 120, line: 340 },
            indent: { firstLine: 720 },
            children: [new this.docx.TextRun({
                text: meta.acknowledgments,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * Table of Contents
     */
    buildTableOfContents() {
        const elements = [];

        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Table of Contents',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.TableOfContents("Table of Contents", {
            hyperlink: true,
            headingStyleRange: "1-3"
        }));

        // Instruction to update TOC
        elements.push(new this.docx.Paragraph({
            spacing: { before: 240, after: 120 },
            children: [new this.docx.TextRun({
                text: '[Right-click here and select "Update Field" to refresh page numbers]',
                italics: true,
                size: this.sizes.small,
                color: '666666',
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * List of Figures
     */
    buildListOfFigures(chapters) {
        const elements = [];

        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'List of Figures',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Note about updating page numbers
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Note: Update page numbers by pressing Ctrl+A then F9 in Microsoft Word.',
                size: this.sizes.small,
                italics: true,
                font: 'Times New Roman',
                color: this.colors.gray
            })]
        }));

        let figNum = 0;
        chapters.forEach(chapter => {
            if (chapter && chapter.sections) {
                chapter.sections.forEach(section => {
                    if (section.figures) {
                        section.figures.forEach(fig => {
                            figNum++;
                            const chapterNum = chapter.chapter_number || '';
                            elements.push(new this.docx.Paragraph({
                                spacing: { after: 120, line: 340 },
                                indent: { left: 0, hanging: 0 },
                                children: [
                                    new this.docx.TextRun({
                                        text: `Figure ${fig.number || figNum}`,
                                        bold: true,
                                        size: this.sizes.body,
                                        font: 'Times New Roman'
                                    }),
                                    new this.docx.TextRun({
                                        text: ` ${fig.caption}`,
                                        size: this.sizes.body,
                                        font: 'Times New Roman'
                                    }),
                                    new this.docx.TextRun({
                                        text: ` (Chapter ${chapterNum})`,
                                        size: this.sizes.small,
                                        italics: true,
                                        font: 'Times New Roman',
                                        color: this.colors.gray
                                    })
                                ]
                            }));
                        });
                    }
                });
            }
        });

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * List of Tables
     */
    buildListOfTables(chapters) {
        const elements = [];

        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'List of Tables',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Note about updating page numbers
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Note: Update page numbers by pressing Ctrl+A then F9 in Microsoft Word.',
                size: this.sizes.small,
                italics: true,
                font: 'Times New Roman',
                color: this.colors.gray
            })]
        }));

        let tableNum = 0;
        chapters.forEach(chapter => {
            if (chapter && chapter.sections) {
                chapter.sections.forEach(section => {
                    if (section.tables) {
                        section.tables.forEach(table => {
                            tableNum++;
                            const chapterNum = chapter.chapter_number || '';
                            elements.push(new this.docx.Paragraph({
                                spacing: { after: 120, line: 340 },
                                indent: { left: 0, hanging: 0 },
                                children: [
                                    new this.docx.TextRun({
                                        text: `Table ${table.number || tableNum}`,
                                        bold: true,
                                        size: this.sizes.body,
                                        font: 'Times New Roman'
                                    }),
                                    new this.docx.TextRun({
                                        text: ` ${table.caption}`,
                                        size: this.sizes.body,
                                        font: 'Times New Roman'
                                    }),
                                    new this.docx.TextRun({
                                        text: ` (Chapter ${chapterNum})`,
                                        size: this.sizes.small,
                                        italics: true,
                                        font: 'Times New Roman',
                                        color: this.colors.gray
                                    })
                                ]
                            }));
                        });
                    }
                });
            }
        });

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * Abstract Page (APA Style)
     */
    buildAbstract(meta) {
        const elements = [];

        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Abstract',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Abstract text - no first line indent per APA
        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.LEFT,
            spacing: { after: 120, line: 340 },
            indent: { firstLine: 0 },  // Explicitly no indent for abstract
            children: [new this.docx.TextRun({
                text: meta.abstract,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Keywords (APA style - italicized label, indented)
        if (meta.keywords) {
            elements.push(new this.docx.Paragraph({
                spacing: { after: 120, line: 340 },
                indent: { firstLine: 720 },
                children: [
                    new this.docx.TextRun({
                        text: 'Keywords: ',
                        italics: true,
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    }),
                    new this.docx.TextRun({
                        text: meta.keywords.join(', '),
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })
                ]
            }));
        }

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * Build Chapter (APA Style)
     */
    async buildChapter(chapter) {
        const elements = [];

        // Chapter title - APA Level 1 (Centered, Bold)
        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: `Chapter ${chapter.chapter_number}: ${chapter.title}`,
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Introduction
        if (chapter.introduction) {
            elements.push(...this.parseHtml(chapter.introduction));
        }

        // Sections
        if (chapter.sections) {
            for (const section of chapter.sections) {
                elements.push(...await this.buildSection(section));
            }
        }

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * Build Section (APA Level 2)
     */
    async buildSection(section) {
        const elements = [];

        // Section heading - APA Level 2 (Left-aligned, Bold)
        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_2,
            alignment: this.docx.AlignmentType.LEFT,
            spacing: { before: 180, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: `${section.section_key} ${section.title}`,
                bold: true,
                size: this.sizes.heading2,
                font: 'Times New Roman'
            })]
        }));

        // Content
        if (section.content_html) {
            elements.push(...this.parseHtml(section.content_html));
        }

        // Tables first (APA places tables near their mention)
        if (section.tables) {
            for (const table of section.tables) {
                elements.push(...this.buildAPATable(table));
            }
        }

        // Figures
        if (section.figures) {
            for (const fig of section.figures) {
                elements.push(...await this.buildAPAFigure(fig));
            }
        }

        // Subsections
        if (section.subsections) {
            for (const sub of section.subsections) {
                elements.push(...await this.buildSubsection(sub));
            }
        }

        return elements;
    },

    /**
     * Build Subsection (APA Level 3)
     */
    async buildSubsection(subsection) {
        const elements = [];

        // APA Level 3 (Left-aligned, Bold Italic)
        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_3,
            alignment: this.docx.AlignmentType.LEFT,
            spacing: { before: 240, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: `${subsection.section_key} ${subsection.title}`,
                bold: true,
                italics: true,
                size: this.sizes.heading3,
                font: 'Times New Roman'
            })]
        }));

        if (subsection.content_html) {
            elements.push(...this.parseHtml(subsection.content_html));
        }

        if (subsection.tables) {
            for (const table of subsection.tables) {
                elements.push(...this.buildAPATable(table));
            }
        }

        if (subsection.figures) {
            for (const fig of subsection.figures) {
                elements.push(...await this.buildAPAFigure(fig));
            }
        }

        return elements;
    },

    /**
     * Build APA Style Figure
     */
    async buildAPAFigure(fig) {
        const elements = [];
        this.figureNumber++;

        elements.push(new this.docx.Paragraph({ spacing: { before: 180 } }));

        // Try to load and embed image
        try {
            const imgPath = fig.src.startsWith('figures/') ? fig.src : `figures/${fig.src}`;
            const response = await fetch(imgPath);

            if (response.ok) {
                const blob = await response.blob();
                const arrayBuffer = await blob.arrayBuffer();
                const uint8Array = new Uint8Array(arrayBuffer);

                elements.push(new this.docx.Paragraph({
                    alignment: this.docx.AlignmentType.CENTER,
                    spacing: { after: 120 },
                    children: [
                        new this.docx.ImageRun({
                            data: uint8Array,
                            transformation: { width: 400, height: 250 },  // Proper size with aspect ratio
                            type: 'png'
                        })
                    ]
                }));
            }
        } catch (e) {
            // Skip image if can't load
        }

        // APA 7 Figure number (bold, on its own line)
        elements.push(new this.docx.Paragraph({
            spacing: { after: 0, line: 340 },
            children: [
                new this.docx.TextRun({
                    text: `Figure ${fig.number || this.figureNumber}`,
                    bold: true,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })
            ]
        }));

        // APA 7 Figure caption (italicized, on its own line)
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 340 },
            children: [
                new this.docx.TextRun({
                    text: fig.caption,
                    italics: true,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })
            ]
        }));

        // Description as Note
        if (fig.description) {
            elements.push(new this.docx.Paragraph({
                alignment: this.docx.AlignmentType.LEFT,
                spacing: { after: 180, line: 340 },
                children: [
                    new this.docx.TextRun({
                        text: 'Note. ',
                        italics: true,
                        size: this.sizes.small,
                        font: 'Times New Roman'
                    }),
                    new this.docx.TextRun({
                        text: fig.description,
                        size: this.sizes.small,
                        font: 'Times New Roman'
                    })
                ]
            }));
        }

        return elements;
    },

    /**
     * Build APA Style Table
     */
    buildAPATable(tableData) {
        const elements = [];
        this.tableNumber++;

        elements.push(new this.docx.Paragraph({ spacing: { before: 180 } }));

        // APA 7 Table number (bold, on its own line)
        elements.push(new this.docx.Paragraph({
            spacing: { after: 0, line: 340 },
            children: [new this.docx.TextRun({
                text: `Table ${tableData.number || this.tableNumber}`,
                bold: true,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // APA 7 Table title (italicized, on its own line)
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: tableData.caption,
                italics: true,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Build table
        const rows = [];

        // Header row - compact 10pt font with cell padding
        if (tableData.headers) {
            rows.push(new this.docx.TableRow({
                tableHeader: true,
                children: tableData.headers.map(header =>
                    new this.docx.TableCell({
                        children: [new this.docx.Paragraph({
                            spacing: { before: 60, after: 60, line: 240 },
                            children: [new this.docx.TextRun({
                                text: header,
                                bold: true,
                                size: this.sizes.table,
                                font: 'Times New Roman'
                            })]
                        })],
                        margins: {
                            top: 40,
                            bottom: 40,
                            left: 80,
                            right: 80
                        },
                        borders: {
                            top: { style: this.docx.BorderStyle.SINGLE, size: 8, color: this.colors.black },
                            bottom: { style: this.docx.BorderStyle.SINGLE, size: 4, color: this.colors.black },
                            left: { style: this.docx.BorderStyle.NONE },
                            right: { style: this.docx.BorderStyle.NONE }
                        }
                    })
                )
            }));
        }

        // Data rows - compact 10pt font with cell padding
        if (tableData.rows) {
            tableData.rows.forEach((row, idx) => {
                const isLast = idx === tableData.rows.length - 1;
                rows.push(new this.docx.TableRow({
                    children: row.map(cell =>
                        new this.docx.TableCell({
                            children: [new this.docx.Paragraph({
                                spacing: { before: 40, after: 40, line: 240 },
                                children: [new this.docx.TextRun({
                                    text: String(cell),
                                    size: this.sizes.table,
                                    font: 'Times New Roman'
                                })]
                            })],
                            margins: {
                                top: 30,
                                bottom: 30,
                                left: 80,
                                right: 80
                            },
                            borders: {
                                top: { style: this.docx.BorderStyle.NONE },
                                bottom: isLast
                                    ? { style: this.docx.BorderStyle.SINGLE, size: 8, color: this.colors.black }
                                    : { style: this.docx.BorderStyle.NONE },
                                left: { style: this.docx.BorderStyle.NONE },
                                right: { style: this.docx.BorderStyle.NONE }
                            }
                        })
                    )
                }));
            });
        }

        elements.push(new this.docx.Table({
            width: { size: 100, type: this.docx.WidthType.PERCENTAGE },
            rows: rows
        }));

        // Note (description)
        if (tableData.description) {
            elements.push(new this.docx.Paragraph({
                alignment: this.docx.AlignmentType.LEFT,
                spacing: { before: 120, after: 180, line: 340 },
                children: [
                    new this.docx.TextRun({
                        text: 'Note. ',
                        italics: true,
                        size: this.sizes.small,
                        font: 'Times New Roman'
                    }),
                    new this.docx.TextRun({
                        text: tableData.description,
                        size: this.sizes.small,
                        font: 'Times New Roman'
                    })
                ]
            }));
        }

        return elements;
    },

    /**
     * Build References (APA Hanging Indent)
     */
    buildReferences(references) {
        const elements = [];

        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'References',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        references.forEach(ref => {
            let text = ref.citation;
            if (ref.url) text += ` ${ref.url}`;
            if (ref.doi) text += ` https://doi.org/${ref.doi}`;

            // APA hanging indent: 0.5 inch (720 twips)
            elements.push(new this.docx.Paragraph({
                spacing: { after: 120, line: 340 },
                indent: { left: 720, hanging: 720 },
                children: [new this.docx.TextRun({
                    text: text,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
        });

        elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        return elements;
    },

    /**
     * Build Appendices with pre-rendered notebook images
     * Uses high-quality screenshots generated by generate_notebook_images.py
     */
    async buildAppendices() {
        const elements = [];

        // Appendices heading
        elements.push(new this.docx.Paragraph({
            heading: this.docx.HeadingLevel.HEADING_1,
            alignment: this.docx.AlignmentType.CENTER,
            spacing: { before: 0, after: 120, line: 340 },
            children: [new this.docx.TextRun({
                text: 'Appendices',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.LEFT,
            spacing: { after: 180, line: 340 },
            indent: { firstLine: 720 },
            children: [new this.docx.TextRun({
                text: 'The following appendices contain the complete Jupyter Notebook implementations for each machine learning model evaluated in this study. Each appendix preserves the exact notebook format with syntax-highlighted code cells, outputs, and visualizations as executed during the experimental phase.',
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Try to load pre-rendered images manifest
        let manifest = null;
        try {
            const response = await fetch('notebook-images/manifest.json');
            if (response.ok) {
                manifest = await response.json();
            }
        } catch (e) {
            console.log('Manifest not found, will use fallback rendering');
        }

        const notebooks = [
            {
                id: 'A',
                key: 'Ensemble',
                title: 'Ensemble Methods Implementation',
                subtitle: 'BaggingClassifier with Decision Tree Base Estimators',
                file: 'notebooks/skML-complete_Ensemble.ipynb'
            },
            {
                id: 'B',
                key: 'NonLinear',
                title: 'Non-Linear Methods Implementation',
                subtitle: 'DecisionTreeClassifier with Optimized Parameters',
                file: 'notebooks/skML-complete_NonLinear.ipynb'
            },
            {
                id: 'C',
                key: 'SupportVector',
                title: 'Linear Methods Implementation',
                subtitle: 'RidgeClassifier with Balanced Class Weights',
                file: 'notebooks/skML-complete_SupportVector.ipynb'
            }
        ];

        for (const nb of notebooks) {
            // Appendix heading (APA Level 2)
            elements.push(new this.docx.Paragraph({
                heading: this.docx.HeadingLevel.HEADING_2,
                alignment: this.docx.AlignmentType.LEFT,
                spacing: { before: 240, after: 120, line: 340 },
                children: [new this.docx.TextRun({
                    text: `Appendix ${nb.id}: ${nb.title}`,
                    bold: true,
                    size: this.sizes.heading2,
                    font: 'Times New Roman'
                })]
            }));

            elements.push(new this.docx.Paragraph({
                spacing: { after: 120, line: 340 },
                children: [new this.docx.TextRun({
                    text: nb.subtitle,
                    italics: true,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));

            // Check if pre-rendered images exist
            if (manifest && manifest[nb.key] && manifest[nb.key].pages) {
                elements.push(...await this.embedNotebookImages(nb.key, manifest[nb.key]));
            } else {
                // Fallback to basic notebook rendering
                try {
                    const response = await fetch(nb.file);
                    if (response.ok) {
                        const notebook = await response.json();
                        elements.push(...await this.renderJupyterNotebook(notebook));
                    } else {
                        elements.push(this.createNotebookFallbackMessage(nb.file));
                    }
                } catch (e) {
                    elements.push(this.createNotebookFallbackMessage(nb.file));
                }
            }

            // Page break after each appendix
            elements.push(new this.docx.Paragraph({ children: [new this.docx.PageBreak()] }));
        }

        return elements;
    },

    /**
     * Embed pre-rendered notebook page images (A4 PDF pages)
     * Each image is one full page from the notebook PDF
     * Sized to fit within 1" margins on US Letter paper
     */
    async embedNotebookImages(notebookKey, manifestData) {
        const elements = [];
        const basePath = 'notebook-images/';

        // Source file reference with better styling
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 340 },
            children: [
                new this.docx.TextRun({
                    text: 'Source: ',
                    bold: true,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                }),
                new this.docx.TextRun({
                    text: manifestData.file,
                    size: this.sizes.body,
                    font: 'Consolas'
                })
            ]
        }));

        elements.push(new this.docx.Paragraph({
            spacing: { after: 180, line: 340 },
            children: [
                new this.docx.TextRun({
                    text: `Total Pages: ${manifestData.pages.length}`,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })
            ]
        }));

        for (let i = 0; i < manifestData.pages.length; i++) {
            const imagePath = basePath + manifestData.pages[i];

            try {
                const response = await fetch(imagePath);
                if (response.ok) {
                    const blob = await response.blob();
                    const arrayBuffer = await blob.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);

                    // Page header - clearer formatting
                    elements.push(new this.docx.Paragraph({
                        alignment: this.docx.AlignmentType.CENTER,
                        spacing: { before: 120, after: 120 },
                        children: [new this.docx.TextRun({
                            text: `— Notebook Page ${i + 1} of ${manifestData.pages.length} —`,
                            size: this.sizes.small,
                            bold: true,
                            font: 'Times New Roman'
                        })]
                    }));

                    // Notebook page image in a table cell with border (creates frame effect)
                    // Size: 6" wide x 8" tall (fits within 1" margins with room for header)
                    const imageTable = new this.docx.Table({
                        width: { size: 100, type: this.docx.WidthType.PERCENTAGE },
                        rows: [
                            new this.docx.TableRow({
                                children: [
                                    new this.docx.TableCell({
                                        children: [
                                            new this.docx.Paragraph({
                                                alignment: this.docx.AlignmentType.CENTER,
                                                spacing: { before: 0, after: 0 },
                                                children: [
                                                    new this.docx.ImageRun({
                                                        data: uint8Array,
                                                        transformation: {
                                                            width: 460,  // ~6.4" fits within margins
                                                            height: 595  // Maintains A4 aspect ratio
                                                        },
                                                        type: 'png'
                                                    })
                                                ]
                                            })
                                        ],
                                        borders: {
                                            top: { style: this.docx.BorderStyle.SINGLE, size: 8, color: '666666' },
                                            bottom: { style: this.docx.BorderStyle.SINGLE, size: 8, color: '666666' },
                                            left: { style: this.docx.BorderStyle.SINGLE, size: 8, color: '666666' },
                                            right: { style: this.docx.BorderStyle.SINGLE, size: 8, color: '666666' }
                                        },
                                        shading: { fill: 'FFFFFF' }
                                    })
                                ]
                            })
                        ]
                    });

                    elements.push(imageTable);

                    // Page break after each image (except the last one)
                    if (i < manifestData.pages.length - 1) {
                        elements.push(new this.docx.Paragraph({
                            spacing: { before: 120 },
                            children: [new this.docx.PageBreak()]
                        }));
                    }
                }
            } catch (e) {
                console.warn(`Failed to load: ${imagePath}`);
            }
        }

        return elements;
    },

    /**
     * Create fallback message for missing notebooks
     */
    createNotebookFallbackMessage(file) {
        return new this.docx.Paragraph({
            spacing: { after: 120, line: 340 },
            shading: { fill: 'FFF3CD' },
            children: [new this.docx.TextRun({
                text: `Note: Pre-rendered notebook images not found. Run 'python generate_notebook_images.py' to generate high-quality notebook images for this appendix. Source file: ${file}`,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        });
    },

    /**
     * Render Jupyter Notebook - Professional format with cell numbers and borders
     */
    async renderJupyterNotebook(notebook) {
        const elements = [];

        if (!notebook.cells) return elements;

        let cellIndex = 0;
        let codeIndex = 0;

        for (const cell of notebook.cells) {
            cellIndex++;

            if (cell.cell_type === 'markdown') {
                // Markdown cells - render as formatted text
                const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
                if (source.trim()) {
                    elements.push(...this.renderMarkdownCell(source));
                }
            } else if (cell.cell_type === 'code') {
                codeIndex++;
                // Code cells - professional Jupyter format
                const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
                const execCount = cell.execution_count || codeIndex;

                if (source.trim()) {
                    // Cell separator
                    elements.push(new this.docx.Paragraph({
                        spacing: { before: 180, after: 120 },
                        children: [new this.docx.TextRun({
                            text: `━━━ Cell ${codeIndex} ━━━`,
                            size: this.sizes.small,
                            color: '999999',
                            font: 'Consolas'
                        })]
                    }));

                    // Input header with blue background
                    elements.push(new this.docx.Paragraph({
                        spacing: { before: 0, after: 0 },
                        shading: { fill: 'E8F4FD' },
                        border: {
                            top: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' },
                            left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' },
                            right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' }
                        },
                        children: [
                            new this.docx.TextRun({
                                text: `In [${execCount}]:`,
                                bold: true,
                                size: this.sizes.code,
                                font: 'Consolas',
                                color: '2980B9'
                            })
                        ]
                    }));

                    // Code content with line numbers in a bordered box
                    const codeLines = source.split('\n');
                    const maxLines = 150; // Increased limit
                    const linesToShow = Math.min(codeLines.length, maxLines);

                    for (let i = 0; i < linesToShow; i++) {
                        const lineNum = String(i + 1).padStart(3, ' ');
                        const isLast = i === linesToShow - 1 && codeLines.length <= maxLines;
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            shading: { fill: 'F8F8F8' },
                            border: {
                                left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' },
                                right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' },
                                bottom: isLast ? { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' } : { style: this.docx.BorderStyle.NONE }
                            },
                            children: [
                                new this.docx.TextRun({
                                    text: `${lineNum} │ `,
                                    size: this.sizes.code,
                                    font: 'Consolas',
                                    color: '999999'
                                }),
                                new this.docx.TextRun({
                                    text: codeLines[i] || ' ',
                                    size: this.sizes.code,
                                    font: 'Consolas'
                                })
                            ]
                        }));
                    }

                    if (codeLines.length > maxLines) {
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            shading: { fill: 'FFF3CD' },
                            border: {
                                left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' },
                                right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' },
                                bottom: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '3498DB' }
                            },
                            children: [new this.docx.TextRun({
                                text: `    ... [${codeLines.length - maxLines} more lines truncated]`,
                                size: this.sizes.code,
                                font: 'Consolas',
                                italics: true,
                                color: '856404'
                            })]
                        }));
                    }

                    // Outputs with different styling
                    if (cell.outputs && cell.outputs.length > 0) {
                        for (const output of cell.outputs) {
                            elements.push(...await this.renderCellOutput(output, execCount));
                        }
                    }
                }
            }
        }

        return elements;
    },

    /**
     * Render markdown cell content with inline formatting support
     */
    renderMarkdownCell(source) {
        const elements = [];
        const lines = source.split('\n');

        for (const line of lines) {
            if (line.startsWith('# ')) {
                // H1
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 180, after: 120, line: 340 },
                    children: this.parseInlineMarkdown(line.substring(2), { bold: true, size: 28 })
                }));
            } else if (line.startsWith('## ')) {
                // H2
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 240, after: 120, line: 340 },
                    children: this.parseInlineMarkdown(line.substring(3), { bold: true, size: 26 })
                }));
            } else if (line.startsWith('### ')) {
                // H3
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 180, after: 120, line: 340 },
                    children: this.parseInlineMarkdown(line.substring(4), { bold: true, italics: true, size: this.sizes.body })
                }));
            } else if (line.startsWith('#### ')) {
                // H4
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 120, after: 60, line: 340 },
                    children: this.parseInlineMarkdown(line.substring(5), { bold: true, size: this.sizes.body })
                }));
            } else if (line.startsWith('- ') || line.startsWith('* ')) {
                // Bullet
                elements.push(new this.docx.Paragraph({
                    numbering: { reference: 'bullet-list', level: 0 },
                    spacing: { after: 60, line: 340 },
                    children: this.parseInlineMarkdown(line.substring(2), { size: this.sizes.body })
                }));
            } else if (line.startsWith('  - ') || line.startsWith('  * ')) {
                // Nested bullet
                elements.push(new this.docx.Paragraph({
                    numbering: { reference: 'bullet-list', level: 1 },
                    spacing: { after: 60, line: 340 },
                    children: this.parseInlineMarkdown(line.substring(4), { size: this.sizes.body })
                }));
            } else if (line.match(/^\d+\.\s/)) {
                // Numbered list
                elements.push(new this.docx.Paragraph({
                    numbering: { reference: 'numbered-list', level: 0 },
                    spacing: { after: 60, line: 340 },
                    children: this.parseInlineMarkdown(line.replace(/^\d+\.\s/, ''), { size: this.sizes.body })
                }));
            } else if (line.startsWith('> ')) {
                // Blockquote
                elements.push(new this.docx.Paragraph({
                    spacing: { after: 120, line: 340 },
                    indent: { left: 720 },
                    shading: { fill: 'F5F5F5' },
                    border: {
                        left: { style: this.docx.BorderStyle.SINGLE, size: 12, color: 'CCCCCC' }
                    },
                    children: this.parseInlineMarkdown(line.substring(2), { size: this.sizes.body, italics: true })
                }));
            } else if (line.startsWith('```')) {
                // Skip code block markers (handled separately)
                continue;
            } else if (line.trim()) {
                // Regular paragraph with inline formatting
                elements.push(new this.docx.Paragraph({
                    spacing: { after: 120, line: 340 },
                    children: this.parseInlineMarkdown(line, { size: this.sizes.body })
                }));
            }
        }

        return elements;
    },

    /**
     * Parse inline markdown formatting (bold, italic, code, links)
     */
    parseInlineMarkdown(text, baseStyle = {}) {
        const runs = [];
        // Pattern to match **bold**, *italic*, `code`, and combinations
        const pattern = /(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|_(.+?)_)/g;

        let lastIndex = 0;
        let match;

        while ((match = pattern.exec(text)) !== null) {
            // Add text before the match
            if (match.index > lastIndex) {
                runs.push(new this.docx.TextRun({
                    text: text.substring(lastIndex, match.index),
                    font: 'Times New Roman',
                    ...baseStyle
                }));
            }

            // Determine the type of formatting
            if (match[2]) {
                // ***bold italic***
                runs.push(new this.docx.TextRun({
                    text: match[2],
                    font: 'Times New Roman',
                    bold: true,
                    italics: true,
                    ...baseStyle
                }));
            } else if (match[3]) {
                // **bold**
                runs.push(new this.docx.TextRun({
                    text: match[3],
                    font: 'Times New Roman',
                    bold: true,
                    ...baseStyle
                }));
            } else if (match[4]) {
                // *italic*
                runs.push(new this.docx.TextRun({
                    text: match[4],
                    font: 'Times New Roman',
                    italics: true,
                    ...baseStyle
                }));
            } else if (match[5]) {
                // `code`
                runs.push(new this.docx.TextRun({
                    text: match[5],
                    font: 'Consolas',
                    size: this.sizes.small,
                    shading: { fill: 'F0F0F0' }
                }));
            } else if (match[6]) {
                // _italic_
                runs.push(new this.docx.TextRun({
                    text: match[6],
                    font: 'Times New Roman',
                    italics: true,
                    ...baseStyle
                }));
            }

            lastIndex = match.index + match[0].length;
        }

        // Add remaining text
        if (lastIndex < text.length) {
            runs.push(new this.docx.TextRun({
                text: text.substring(lastIndex),
                font: 'Times New Roman',
                ...baseStyle
            }));
        }

        // If no matches were found, return the whole text
        if (runs.length === 0) {
            runs.push(new this.docx.TextRun({
                text: text,
                font: 'Times New Roman',
                ...baseStyle
            }));
        }

        return runs;
    },

    /**
     * Render cell output - including images with improved styling
     */
    async renderCellOutput(output, execCount) {
        const elements = [];
        const maxOutputLines = 100; // Increased from 30

        // Handle different output types
        if (output.output_type === 'stream') {
            // Stream output (print statements) with green styling
            const text = Array.isArray(output.text) ? output.text.join('') : (output.text || '');
            if (text.trim()) {
                // Output header
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 120, after: 0 },
                    shading: { fill: 'E8F5E9' },
                    border: {
                        top: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' },
                        left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' },
                        right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' }
                    },
                    children: [new this.docx.TextRun({
                        text: output.name === 'stderr' ? 'stderr:' : 'Output:',
                        bold: true,
                        size: this.sizes.code,
                        font: 'Consolas',
                        color: output.name === 'stderr' ? 'C0392B' : '27AE60'
                    })]
                }));

                const allLines = text.split('\n');
                const lines = allLines.slice(0, maxOutputLines);
                for (let i = 0; i < lines.length; i++) {
                    const isLast = i === lines.length - 1 && allLines.length <= maxOutputLines;
                    elements.push(new this.docx.Paragraph({
                        spacing: { after: 0 },
                        shading: { fill: 'FAFAFA' },
                        border: {
                            left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' },
                            right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' },
                            bottom: isLast ? { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' } : { style: this.docx.BorderStyle.NONE }
                        },
                        indent: { left: 180 },
                        children: [new this.docx.TextRun({
                            text: lines[i] || ' ',
                            size: this.sizes.code,
                            font: 'Consolas'
                        })]
                    }));
                }

                if (allLines.length > maxOutputLines) {
                    elements.push(new this.docx.Paragraph({
                        spacing: { after: 0 },
                        shading: { fill: 'FFF3CD' },
                        border: {
                            left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' },
                            right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' },
                            bottom: { style: this.docx.BorderStyle.SINGLE, size: 4, color: '27AE60' }
                        },
                        indent: { left: 180 },
                        children: [new this.docx.TextRun({
                            text: `... [${allLines.length - maxOutputLines} more lines truncated]`,
                            size: this.sizes.code,
                            font: 'Consolas',
                            italics: true,
                            color: '856404'
                        })]
                    }));
                }
            }
        } else if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
            // Check for image output first
            if (output.data && output.data['image/png']) {
                try {
                    // Decode base64 image
                    const base64Data = output.data['image/png'];
                    const binaryString = atob(base64Data);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }

                    // Output label
                    elements.push(new this.docx.Paragraph({
                        spacing: { before: 120, after: 60 },
                        children: [
                            new this.docx.TextRun({
                                text: `Out[${execCount}]:`,
                                bold: true,
                                size: this.sizes.code,
                                font: 'Consolas',
                                color: 'AA0000'
                            })
                        ]
                    }));

                    // Embed the image
                    elements.push(new this.docx.Paragraph({
                        alignment: this.docx.AlignmentType.CENTER,
                        spacing: { before: 120, after: 120 },
                        children: [
                            new this.docx.ImageRun({
                                data: bytes,
                                transformation: { width: 500, height: 350 },
                                type: 'png'
                            })
                        ]
                    }));
                } catch (e) {
                    // If image fails, show placeholder
                    elements.push(new this.docx.Paragraph({
                        spacing: { after: 120 },
                        indent: { left: 360 },
                        children: [new this.docx.TextRun({
                            text: '[Output Image]',
                            italics: true,
                            size: this.sizes.code,
                            font: 'Consolas',
                            color: '888888'
                        })]
                    }));
                }
            } else if (output.data && output.data['text/plain']) {
                // Text output with red styling for Out[]
                const text = Array.isArray(output.data['text/plain'])
                    ? output.data['text/plain'].join('')
                    : output.data['text/plain'];

                if (text.trim()) {
                    // Output header
                    elements.push(new this.docx.Paragraph({
                        spacing: { before: 120, after: 0 },
                        shading: { fill: 'FDEDEC' },
                        border: {
                            top: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' },
                            left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' },
                            right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' }
                        },
                        children: [
                            new this.docx.TextRun({
                                text: `Out[${execCount}]:`,
                                bold: true,
                                size: this.sizes.code,
                                font: 'Consolas',
                                color: 'C0392B'
                            })
                        ]
                    }));

                    const allLines = text.split('\n');
                    const maxLines = 80;
                    const lines = allLines.slice(0, maxLines);
                    for (let i = 0; i < lines.length; i++) {
                        const isLast = i === lines.length - 1 && allLines.length <= maxLines;
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            shading: { fill: 'FAFAFA' },
                            border: {
                                left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' },
                                right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' },
                                bottom: isLast ? { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' } : { style: this.docx.BorderStyle.NONE }
                            },
                            indent: { left: 180 },
                            children: [new this.docx.TextRun({
                                text: lines[i] || ' ',
                                size: this.sizes.code,
                                font: 'Consolas'
                            })]
                        }));
                    }

                    if (allLines.length > maxLines) {
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            shading: { fill: 'FFF3CD' },
                            border: {
                                left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' },
                                right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' },
                                bottom: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'E74C3C' }
                            },
                            indent: { left: 180 },
                            children: [new this.docx.TextRun({
                                text: `... [${allLines.length - maxLines} more lines truncated]`,
                                size: this.sizes.code,
                                font: 'Consolas',
                                italics: true,
                                color: '856404'
                            })]
                        }));
                    }
                }
            }
        } else if (output.output_type === 'error') {
            // Error output with prominent red styling
            elements.push(new this.docx.Paragraph({
                spacing: { before: 120, after: 0 },
                shading: { fill: 'FADBD8' },
                border: {
                    top: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'C0392B' },
                    left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'C0392B' },
                    right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'C0392B' }
                },
                children: [new this.docx.TextRun({
                    text: `Error: ${output.ename || 'Exception'}`,
                    bold: true,
                    size: this.sizes.code,
                    font: 'Consolas',
                    color: 'C0392B'
                })]
            }));

            elements.push(new this.docx.Paragraph({
                spacing: { after: 0 },
                shading: { fill: 'FFF5F5' },
                border: {
                    left: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'C0392B' },
                    right: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'C0392B' },
                    bottom: { style: this.docx.BorderStyle.SINGLE, size: 4, color: 'C0392B' }
                },
                indent: { left: 180 },
                children: [new this.docx.TextRun({
                    text: output.evalue || 'An error occurred',
                    size: this.sizes.code,
                    font: 'Consolas',
                    color: 'C0392B'
                })]
            }));
        }

        return elements;
    },

    /**
     * Parse HTML content
     */
    parseHtml(html) {
        const elements = [];
        const temp = document.createElement('div');
        temp.innerHTML = html;

        temp.childNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                elements.push(...this.processElement(node));
            } else if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
                elements.push(new this.docx.Paragraph({
                    alignment: this.docx.AlignmentType.LEFT,
                    spacing: { after: 120, line: 340 },
                    indent: { firstLine: 720 },
                    children: [new this.docx.TextRun({
                        text: node.textContent.trim(),
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
            }
        });

        return elements;
    },

    /**
     * Process HTML element
     */
    processElement(element) {
        const elements = [];
        const tag = element.tagName.toLowerCase();

        switch (tag) {
            case 'p':
                elements.push(new this.docx.Paragraph({
                    alignment: this.docx.AlignmentType.LEFT,
                    spacing: { after: 120, line: 340 },
                    indent: { firstLine: 720 },
                    children: this.getTextRuns(element)
                }));
                break;

            case 'ul':
                element.querySelectorAll(':scope > li').forEach(li => {
                    elements.push(new this.docx.Paragraph({
                        numbering: { reference: 'bullet-list', level: 0 },
                        spacing: { after: 120, line: 340 },
                        children: this.getTextRuns(li)
                    }));
                });
                break;

            case 'ol':
                element.querySelectorAll(':scope > li').forEach(li => {
                    elements.push(new this.docx.Paragraph({
                        numbering: { reference: 'numbered-list', level: 0 },
                        spacing: { after: 120, line: 340 },
                        children: this.getTextRuns(li)
                    }));
                });
                break;

            case 'h3':
            case 'h4':
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 240, after: 120, line: 340 },
                    children: [new this.docx.TextRun({
                        text: element.textContent,
                        bold: true,
                        italics: tag === 'h4',
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
                break;

            case 'blockquote':
                elements.push(new this.docx.Paragraph({
                    indent: { left: 720 },
                    spacing: { after: 120, line: 340 },
                    children: [new this.docx.TextRun({
                        text: element.textContent,
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
                break;

            default:
                if (element.textContent.trim()) {
                    elements.push(new this.docx.Paragraph({
                        alignment: this.docx.AlignmentType.LEFT,
                        spacing: { after: 120, line: 340 },
                        children: [new this.docx.TextRun({
                            text: element.textContent.trim(),
                            size: this.sizes.body,
                            font: 'Times New Roman'
                        })]
                    }));
                }
        }

        return elements;
    },

    /**
     * Get text runs from element
     */
    getTextRuns(element) {
        const runs = [];

        element.childNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE) {
                const text = node.textContent;
                if (text.trim() || text.includes(' ')) {
                    runs.push(new this.docx.TextRun({
                        text: text,
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    }));
                }
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                const tag = node.tagName.toLowerCase();
                const text = node.textContent;

                switch (tag) {
                    case 'strong':
                    case 'b':
                        runs.push(new this.docx.TextRun({
                            text: text,
                            bold: true,
                            size: this.sizes.body,
                            font: 'Times New Roman'
                        }));
                        break;
                    case 'em':
                    case 'i':
                        runs.push(new this.docx.TextRun({
                            text: text,
                            italics: true,
                            size: this.sizes.body,
                            font: 'Times New Roman'
                        }));
                        break;
                    case 'code':
                        runs.push(new this.docx.TextRun({
                            text: text,
                            font: 'Consolas',
                            size: this.sizes.small
                        }));
                        break;
                    default:
                        runs.push(new this.docx.TextRun({
                            text: text,
                            size: this.sizes.body,
                            font: 'Times New Roman'
                        }));
                }
            }
        });

        return runs;
    }
};

window.ThesisExport = ThesisExport;
