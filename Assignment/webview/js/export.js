/**
 * DACS Thesis Document - APA 7th Edition Academic DOCX Export
 * Professional academic styling with proper notebook embedding
 */

const ThesisExport = {
    docx: null,
    figureNumber: 0,
    tableNumber: 0,
    shortTitle: 'MACHINE LEARNING FOR INTRUSION DETECTION',  // Running header

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

    // Font sizes (in half-points) - APA uses 12pt throughout
    sizes: {
        title: 28,          // 14pt for title
        heading1: 24,       // 12pt
        heading2: 24,       // 12pt
        heading3: 24,       // 12pt
        body: 24,           // 12pt
        caption: 24,        // 12pt (APA captions same size)
        small: 20,          // 10pt for notes
        code: 20,           // 10pt for code
        header: 20          // 10pt for header/footer
    },

    /**
     * Export document to DOCX
     */
    async exportToDocx() {
        if (typeof docx === 'undefined') {
            alert('Export library not available. Please refresh the page.');
            return;
        }

        this.docx = docx;
        this.figureNumber = 0;
        this.tableNumber = 0;

        const btn = document.getElementById('exportBtn');
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
                btn.innerHTML = originalContent;
                btn.disabled = false;
            }, 2000);

        } catch (error) {
            console.error('Export failed:', error);
            btn.innerHTML = originalContent;
            btn.disabled = false;
            alert('Export failed: ' + error.message);
        }
    },

    /**
     * Build the complete document with multiple sections
     */
    async buildDocument(data) {
        const sections = [];

        // Section 1: Title page (no header/footer)
        const titleContent = this.buildTitlePage(data.metadata);
        sections.push({
            properties: this.getSectionProperties({ titlePage: true }),
            children: titleContent
        });

        // Section 2: Front matter (Roman numeral page numbers)
        const frontMatter = [];
        frontMatter.push(...this.buildDeclaration(data.metadata));
        frontMatter.push(...this.buildAcknowledgments(data.metadata));
        frontMatter.push(...this.buildTableOfContents());
        frontMatter.push(...this.buildListOfFigures(data.chapters));
        frontMatter.push(...this.buildListOfTables(data.chapters));
        frontMatter.push(...this.buildAbstract(data.metadata));

        sections.push({
            properties: this.getSectionProperties({
                pageNumberFormat: 'lowerRoman',
                pageNumberStart: 1,
                hasHeader: true,
                headerText: 'DACS Assignment'
            }),
            headers: this.createHeaders('DACS Assignment'),
            footers: this.createFooters('roman'),
            children: frontMatter
        });

        // Section 3: Main content (Arabic numeral page numbers)
        const mainContent = [];
        for (const chapter of data.chapters) {
            mainContent.push(...await this.buildChapter(chapter));
        }
        if (data.references?.references) {
            mainContent.push(...this.buildReferences(data.references.references));
        }
        mainContent.push(...await this.buildAppendices());

        sections.push({
            properties: this.getSectionProperties({
                pageNumberFormat: 'decimal',
                pageNumberStart: 1,
                hasHeader: true,
                headerText: this.shortTitle
            }),
            headers: this.createHeaders(this.shortTitle),
            footers: this.createFooters('arabic'),
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
                    top: 1440,    // 1 inch
                    right: 1440,  // 1 inch
                    bottom: 1440, // 1 inch
                    left: 1440    // 1 inch
                }
            }
        };

        if (options.titlePage) {
            props.titlePage = true;
        }

        return props;
    },

    /**
     * Create headers for a section (APA 7: running head left, page number right)
     */
    createHeaders(text) {
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
                        spacing: { line: 480, after: 0 },  // Double spacing (APA 7)
                        alignment: this.docx.AlignmentType.JUSTIFIED
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
                        spacing: { after: 240, line: 480 }
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
                        spacing: { before: 480, after: 240, line: 480 },
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
                        spacing: { before: 360, after: 240, line: 480 },
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
                        spacing: { before: 240, after: 120, line: 480 },
                        outlineLevel: 2
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
                    levels: [{
                        level: 0,
                        format: this.docx.LevelFormat.BULLET,
                        text: '•',
                        alignment: this.docx.AlignmentType.LEFT,
                        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                    }]
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
            elements.push(new P({ spacing: { after: 240 } }));
        }

        // Title (Bold, Centered)
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 240, line: 480 },
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
                spacing: { after: 480, line: 480 },
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
                spacing: { after: 0, line: 480 },
                children: [new T({
                    text: author.name,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
        });

        elements.push(new P({ spacing: { after: 240 } }));

        // Institution
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 480 },
            children: [new T({
                text: meta.institution,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Module
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 480 },
            children: [new T({
                text: meta.module,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Supervisor
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 480 },
            children: [new T({
                text: `Supervisor: ${meta.supervisor}`,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Date
        elements.push(new P({
            alignment: CENTER,
            spacing: { after: 0, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
            children: [new this.docx.TextRun({
                text: 'Declaration of Originality',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Declaration text
        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.JUSTIFIED,
            spacing: { after: 240, line: 480 },
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
                spacing: { after: 480, line: 480 },
                children: [new this.docx.TextRun({
                    text: '_'.repeat(50),
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })]
            }));
            elements.push(new this.docx.Paragraph({
                spacing: { after: 240, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
            children: [new this.docx.TextRun({
                text: 'Acknowledgments',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.JUSTIFIED,
            spacing: { after: 240, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
            children: [new this.docx.TextRun({
                text: 'List of Figures',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Note about updating page numbers
        elements.push(new this.docx.Paragraph({
            spacing: { after: 240, line: 480 },
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
                                spacing: { after: 120, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
            children: [new this.docx.TextRun({
                text: 'List of Tables',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Note about updating page numbers
        elements.push(new this.docx.Paragraph({
            spacing: { after: 240, line: 480 },
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
                                spacing: { after: 120, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
            children: [new this.docx.TextRun({
                text: 'Abstract',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        // Abstract text - no first line indent per APA
        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.JUSTIFIED,
            spacing: { after: 240, line: 480 },
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
                spacing: { after: 240, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
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
            spacing: { before: 360, after: 240, line: 480 },
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
            spacing: { before: 240, after: 120, line: 480 },
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

        elements.push(new this.docx.Paragraph({ spacing: { before: 360 } }));

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
                    spacing: { after: 240 },
                    children: [
                        new this.docx.ImageRun({
                            data: uint8Array,
                            transformation: { width: 450, height: 280 },
                            type: 'png'
                        })
                    ]
                }));
            }
        } catch (e) {
            // Skip image if can't load
        }

        // APA Figure caption: "Figure X" in italics, then caption
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 480 },
            children: [
                new this.docx.TextRun({
                    text: `Figure ${fig.number || this.figureNumber}`,
                    italics: true,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                })
            ]
        }));

        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 480 },
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
                alignment: this.docx.AlignmentType.JUSTIFIED,
                spacing: { after: 360, line: 480 },
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

        elements.push(new this.docx.Paragraph({ spacing: { before: 360 } }));

        // APA Table number
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 480 },
            children: [new this.docx.TextRun({
                text: `Table ${tableData.number || this.tableNumber}`,
                italics: true,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Table title (italicized)
        elements.push(new this.docx.Paragraph({
            spacing: { after: 120, line: 480 },
            children: [new this.docx.TextRun({
                text: tableData.caption,
                italics: true,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Build table
        const rows = [];

        // Header row
        if (tableData.headers) {
            rows.push(new this.docx.TableRow({
                tableHeader: true,
                children: tableData.headers.map(header =>
                    new this.docx.TableCell({
                        children: [new this.docx.Paragraph({
                            spacing: { after: 0 },
                            children: [new this.docx.TextRun({
                                text: header,
                                bold: true,
                                size: this.sizes.body,
                                font: 'Times New Roman'
                            })]
                        })],
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

        // Data rows
        if (tableData.rows) {
            tableData.rows.forEach((row, idx) => {
                const isLast = idx === tableData.rows.length - 1;
                rows.push(new this.docx.TableRow({
                    children: row.map(cell =>
                        new this.docx.TableCell({
                            children: [new this.docx.Paragraph({
                                spacing: { after: 0 },
                                children: [new this.docx.TextRun({
                                    text: String(cell),
                                    size: this.sizes.body,
                                    font: 'Times New Roman'
                                })]
                            })],
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
                alignment: this.docx.AlignmentType.JUSTIFIED,
                spacing: { before: 120, after: 360, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
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
                spacing: { after: 240, line: 480 },
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
            spacing: { before: 0, after: 240, line: 480 },
            children: [new this.docx.TextRun({
                text: 'Appendices',
                bold: true,
                size: this.sizes.heading1,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new this.docx.Paragraph({
            alignment: this.docx.AlignmentType.JUSTIFIED,
            spacing: { after: 360, line: 480 },
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
                spacing: { before: 480, after: 120, line: 480 },
                children: [new this.docx.TextRun({
                    text: `Appendix ${nb.id}: ${nb.title}`,
                    bold: true,
                    size: this.sizes.heading2,
                    font: 'Times New Roman'
                })]
            }));

            elements.push(new this.docx.Paragraph({
                spacing: { after: 240, line: 480 },
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
     */
    async embedNotebookImages(notebookKey, manifestData) {
        const elements = [];
        const basePath = 'notebook-images/';

        // Source file reference
        elements.push(new this.docx.Paragraph({
            spacing: { after: 200, line: 480 },
            children: [
                new this.docx.TextRun({
                    text: 'Source File: ',
                    bold: true,
                    size: this.sizes.body,
                    font: 'Times New Roman'
                }),
                new this.docx.TextRun({
                    text: manifestData.file,
                    size: this.sizes.body,
                    font: 'Consolas'
                }),
                new this.docx.TextRun({
                    text: ` (${manifestData.pages.length} pages)`,
                    size: this.sizes.body,
                    italics: true,
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

                    // Page header
                    elements.push(new this.docx.Paragraph({
                        alignment: this.docx.AlignmentType.RIGHT,
                        spacing: { before: 0, after: 60 },
                        children: [new this.docx.TextRun({
                            text: `Page ${i + 1} of ${manifestData.pages.length}`,
                            size: 18,
                            italics: true,
                            color: this.colors.gray,
                            font: 'Times New Roman'
                        })]
                    }));

                    // Full-page notebook image (A4 proportions: 210mm x 297mm ≈ 1:1.414)
                    // Width: 6 inches = 540 points, Height: 8.5 inches = 765 points
                    elements.push(new this.docx.Paragraph({
                        alignment: this.docx.AlignmentType.CENTER,
                        spacing: { before: 0, after: 0 },
                        children: [
                            new this.docx.ImageRun({
                                data: uint8Array,
                                transformation: {
                                    width: 540,
                                    height: 765
                                },
                                type: 'png'
                            })
                        ]
                    }));

                    // Page break after each image (except the last one)
                    if (i < manifestData.pages.length - 1) {
                        elements.push(new this.docx.Paragraph({
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
            spacing: { after: 240, line: 480 },
            shading: { fill: 'FFF3CD' },
            children: [new this.docx.TextRun({
                text: `Note: Pre-rendered notebook images not found. Run 'python generate_notebook_images.py' to generate high-quality notebook images for this appendix. Source file: ${file}`,
                size: this.sizes.body,
                font: 'Times New Roman'
            })]
        });
    },

    /**
     * Render Jupyter Notebook - Exact browser-like format
     */
    async renderJupyterNotebook(notebook) {
        const elements = [];

        if (!notebook.cells) return elements;

        let cellIndex = 0;

        for (const cell of notebook.cells) {
            cellIndex++;

            if (cell.cell_type === 'markdown') {
                // Markdown cells - render as formatted text
                const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
                if (source.trim()) {
                    elements.push(...this.renderMarkdownCell(source));
                }
            } else if (cell.cell_type === 'code') {
                // Code cells - exact Jupyter format
                const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
                const execCount = cell.execution_count || ' ';

                if (source.trim()) {
                    // Input cell with bracket notation
                    elements.push(new this.docx.Paragraph({
                        spacing: { before: 240, after: 60 },
                        children: [
                            new this.docx.TextRun({
                                text: `In [${execCount}]:`,
                                bold: true,
                                size: this.sizes.code,
                                font: 'Consolas',
                                color: '0000AA'
                            })
                        ]
                    }));

                    // Code content in a bordered box
                    const codeLines = source.split('\n');
                    for (let i = 0; i < Math.min(codeLines.length, 50); i++) {
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            shading: { fill: 'F7F7F7' },
                            indent: { left: 360 },
                            children: [new this.docx.TextRun({
                                text: codeLines[i] || ' ',
                                size: this.sizes.code,
                                font: 'Consolas'
                            })]
                        }));
                    }

                    if (codeLines.length > 50) {
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            shading: { fill: 'F7F7F7' },
                            indent: { left: 360 },
                            children: [new this.docx.TextRun({
                                text: `... [${codeLines.length - 50} more lines]`,
                                size: this.sizes.code,
                                font: 'Consolas',
                                italics: true,
                                color: '888888'
                            })]
                        }));
                    }

                    // Outputs
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
     * Render markdown cell content
     */
    renderMarkdownCell(source) {
        const elements = [];
        const lines = source.split('\n');

        for (const line of lines) {
            if (line.startsWith('# ')) {
                // H1
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 360, after: 120, line: 480 },
                    children: [new this.docx.TextRun({
                        text: line.substring(2),
                        bold: true,
                        size: 28,
                        font: 'Times New Roman'
                    })]
                }));
            } else if (line.startsWith('## ')) {
                // H2
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 240, after: 120, line: 480 },
                    children: [new this.docx.TextRun({
                        text: line.substring(3),
                        bold: true,
                        size: 26,
                        font: 'Times New Roman'
                    })]
                }));
            } else if (line.startsWith('### ')) {
                // H3
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 180, after: 120, line: 480 },
                    children: [new this.docx.TextRun({
                        text: line.substring(4),
                        bold: true,
                        italics: true,
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
            } else if (line.startsWith('- ') || line.startsWith('* ')) {
                // Bullet
                elements.push(new this.docx.Paragraph({
                    bullet: { level: 0 },
                    spacing: { after: 60, line: 480 },
                    children: [new this.docx.TextRun({
                        text: line.substring(2),
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
            } else if (line.match(/^\d+\.\s/)) {
                // Numbered list
                elements.push(new this.docx.Paragraph({
                    numbering: { reference: 'numbered-list', level: 0 },
                    spacing: { after: 60, line: 480 },
                    children: [new this.docx.TextRun({
                        text: line.replace(/^\d+\.\s/, ''),
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
            } else if (line.trim()) {
                // Regular paragraph
                elements.push(new this.docx.Paragraph({
                    spacing: { after: 120, line: 480 },
                    children: [new this.docx.TextRun({
                        text: line,
                        size: this.sizes.body,
                        font: 'Times New Roman'
                    })]
                }));
            }
        }

        return elements;
    },

    /**
     * Render cell output - including images
     */
    async renderCellOutput(output, execCount) {
        const elements = [];

        // Handle different output types
        if (output.output_type === 'stream') {
            // Stream output (print statements)
            const text = Array.isArray(output.text) ? output.text.join('') : (output.text || '');
            if (text.trim()) {
                const lines = text.split('\n').slice(0, 30);
                for (const line of lines) {
                    elements.push(new this.docx.Paragraph({
                        spacing: { after: 0 },
                        shading: { fill: 'FFFFFF' },
                        indent: { left: 360 },
                        children: [new this.docx.TextRun({
                            text: line || ' ',
                            size: this.sizes.code,
                            font: 'Consolas'
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
                        spacing: { before: 120, after: 240 },
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
                // Text output
                const text = Array.isArray(output.data['text/plain'])
                    ? output.data['text/plain'].join('')
                    : output.data['text/plain'];

                if (text.trim()) {
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

                    const lines = text.split('\n').slice(0, 20);
                    for (const line of lines) {
                        elements.push(new this.docx.Paragraph({
                            spacing: { after: 0 },
                            indent: { left: 360 },
                            children: [new this.docx.TextRun({
                                text: line || ' ',
                                size: this.sizes.code,
                                font: 'Consolas'
                            })]
                        }));
                    }
                }
            }
        } else if (output.output_type === 'error') {
            // Error output
            elements.push(new this.docx.Paragraph({
                spacing: { before: 60, after: 60 },
                shading: { fill: 'FFF0F0' },
                indent: { left: 360 },
                children: [new this.docx.TextRun({
                    text: output.ename ? `${output.ename}: ${output.evalue}` : 'Error',
                    size: this.sizes.code,
                    font: 'Consolas',
                    color: 'CC0000'
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
                    alignment: this.docx.AlignmentType.JUSTIFIED,
                    spacing: { after: 240, line: 480 },
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
                    alignment: this.docx.AlignmentType.JUSTIFIED,
                    spacing: { after: 240, line: 480 },
                    indent: { firstLine: 720 },
                    children: this.getTextRuns(element)
                }));
                break;

            case 'ul':
                element.querySelectorAll(':scope > li').forEach(li => {
                    elements.push(new this.docx.Paragraph({
                        bullet: { level: 0 },
                        spacing: { after: 120, line: 480 },
                        children: this.getTextRuns(li)
                    }));
                });
                break;

            case 'ol':
                element.querySelectorAll(':scope > li').forEach(li => {
                    elements.push(new this.docx.Paragraph({
                        numbering: { reference: 'numbered-list', level: 0 },
                        spacing: { after: 120, line: 480 },
                        children: this.getTextRuns(li)
                    }));
                });
                break;

            case 'h3':
            case 'h4':
                elements.push(new this.docx.Paragraph({
                    spacing: { before: 240, after: 120, line: 480 },
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
                    spacing: { after: 240, line: 480 },
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
                        alignment: this.docx.AlignmentType.JUSTIFIED,
                        spacing: { after: 240, line: 480 },
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
