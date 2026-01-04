#!/usr/bin/env node
/**
 * Node.js DOCX Generator for Testing
 * Generates the same DOCX output as the browser-based export
 */

const fs = require('fs');
const path = require('path');
const { Document, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, HeadingLevel, BorderStyle, WidthType, PageBreak,
        Packer, TabStopType, TabStopPosition, LeaderType, LevelFormat, TableOfContents,
        Header, Footer, PageNumber } = require('docx');

const CONTENT_PATH = './content/';
const FIGURES_PATH = './figures/';
const NOTEBOOK_IMAGES_PATH = './notebook-images/';

// APA Style colors
const colors = {
    black: '000000',
    darkGray: '333333',
    gray: '666666',
    lightGray: '999999',
    tableBorder: '000000',
    tableHeaderBg: 'F2F2F2',
    tableAltRow: 'FAFAFA',
    link: '0000FF'
};

// Font sizes (in half-points)
const sizes = {
    title: 28,
    heading1: 24,
    heading2: 24,
    heading3: 24,
    body: 24,
    caption: 24,
    small: 20,
    code: 20,
    header: 20
};

const SHORT_TITLE = 'MACHINE LEARNING FOR INTRUSION DETECTION';

// Create header (APA 7: running head left, page number right)
function createHeader(text) {
    return new Header({
        children: [
            new Paragraph({
                alignment: AlignmentType.LEFT,
                spacing: { after: 0 },
                tabStops: [{
                    type: TabStopType.RIGHT,
                    position: 9360  // Right margin position
                }],
                children: [
                    new TextRun({
                        text: text,
                        size: sizes.header,
                        font: 'Times New Roman'
                    }),
                    new TextRun({
                        text: '\t'
                    }),
                    new TextRun({
                        children: [PageNumber.CURRENT],
                        size: sizes.header,
                        font: 'Times New Roman'
                    })
                ]
            })
        ]
    });
}

// Create footer with page numbers
function createFooter() {
    return new Footer({
        children: [
            new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                    new TextRun({
                        children: [PageNumber.CURRENT],
                        size: sizes.header,
                        font: 'Times New Roman'
                    })
                ]
            })
        ]
    });
}

let figureNumber = 0;
let tableNumber = 0;

async function loadContent() {
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

    const data = {};

    for (const file of files) {
        const filePath = path.join(CONTENT_PATH, file);
        if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath, 'utf-8');
            const key = file.replace('.json', '');
            data[key] = JSON.parse(content);
        }
    }

    return {
        metadata: data.metadata,
        chapters: [data.chapter1, data.chapter2, data.chapter3, data.chapter4, data.chapter5].filter(Boolean),
        references: data.references,
        annex: data.annex
    };
}

function buildTitlePage(meta) {
    const elements = [];

    // Spacing from top
    for (let i = 0; i < 6; i++) {
        elements.push(new Paragraph({ spacing: { after: 240 } }));
    }

    // Title
    elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 240, line: 480 },
        children: [new TextRun({
            text: meta.title,
            bold: true,
            size: sizes.title,
            font: 'Times New Roman'
        })]
    }));

    // Subtitle
    if (meta.subtitle) {
        elements.push(new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 480, line: 480 },
            children: [new TextRun({
                text: meta.subtitle,
                size: sizes.body,
                font: 'Times New Roman'
            })]
        }));
    }

    // Authors
    meta.authors.forEach(author => {
        elements.push(new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 0, line: 480 },
            children: [new TextRun({
                text: author.name,
                size: sizes.body,
                font: 'Times New Roman'
            })]
        }));
    });

    elements.push(new Paragraph({ spacing: { after: 240 } }));

    // Institution
    elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0, line: 480 },
        children: [new TextRun({
            text: meta.institution,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // Module
    elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0, line: 480 },
        children: [new TextRun({
            text: meta.module,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // Supervisor
    elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0, line: 480 },
        children: [new TextRun({
            text: `Supervisor: ${meta.supervisor}`,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // Date
    elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0, line: 480 },
        children: [new TextRun({
            text: meta.date,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // No page break - sections handle this
    return elements;
}

function buildDeclaration(meta) {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'Declaration of Originality',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 240, line: 480 },
        indent: { firstLine: 720 },
        children: [new TextRun({
            text: meta.declaration,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new Paragraph({ spacing: { after: 480 } }));

    meta.authors.forEach(author => {
        elements.push(new Paragraph({
            spacing: { after: 480, line: 480 },
            children: [new TextRun({
                text: '_'.repeat(50),
                size: sizes.body,
                font: 'Times New Roman'
            })]
        }));
        elements.push(new Paragraph({
            spacing: { after: 240, line: 480 },
            children: [new TextRun({
                text: `${author.name} (${author.id})`,
                size: sizes.body,
                font: 'Times New Roman'
            })]
        }));
    });

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function buildAcknowledgments(meta) {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'Acknowledgments',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 240, line: 480 },
        indent: { firstLine: 720 },
        children: [new TextRun({
            text: meta.acknowledgments,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function buildTableOfContents() {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'Table of Contents',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new TableOfContents("Table of Contents", {
        hyperlink: true,
        headingStyleRange: "1-3"
    }));

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function buildListOfFigures(chapters) {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'List of Figures',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    // Note about updating page numbers
    elements.push(new Paragraph({
        spacing: { after: 240, line: 480 },
        children: [new TextRun({
            text: 'Note: Update page numbers by pressing Ctrl+A then F9 in Microsoft Word.',
            size: sizes.small,
            italics: true,
            font: 'Times New Roman',
            color: colors.gray
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
                        elements.push(new Paragraph({
                            spacing: { after: 120, line: 480 },
                            children: [
                                new TextRun({
                                    text: `Figure ${fig.number || figNum}`,
                                    bold: true,
                                    size: sizes.body,
                                    font: 'Times New Roman'
                                }),
                                new TextRun({
                                    text: ` ${fig.caption}`,
                                    size: sizes.body,
                                    font: 'Times New Roman'
                                }),
                                new TextRun({
                                    text: ` (Chapter ${chapterNum})`,
                                    size: sizes.small,
                                    italics: true,
                                    font: 'Times New Roman',
                                    color: colors.gray
                                })
                            ]
                        }));
                    });
                }
            });
        }
    });

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function buildListOfTables(chapters) {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'List of Tables',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    // Note about updating page numbers
    elements.push(new Paragraph({
        spacing: { after: 240, line: 480 },
        children: [new TextRun({
            text: 'Note: Update page numbers by pressing Ctrl+A then F9 in Microsoft Word.',
            size: sizes.small,
            italics: true,
            font: 'Times New Roman',
            color: colors.gray
        })]
    }));

    let tblNum = 0;
    chapters.forEach(chapter => {
        if (chapter && chapter.sections) {
            chapter.sections.forEach(section => {
                if (section.tables) {
                    section.tables.forEach(table => {
                        tblNum++;
                        const chapterNum = chapter.chapter_number || '';
                        elements.push(new Paragraph({
                            spacing: { after: 120, line: 480 },
                            children: [
                                new TextRun({
                                    text: `Table ${table.number || tblNum}`,
                                    bold: true,
                                    size: sizes.body,
                                    font: 'Times New Roman'
                                }),
                                new TextRun({
                                    text: ` ${table.caption}`,
                                    size: sizes.body,
                                    font: 'Times New Roman'
                                }),
                                new TextRun({
                                    text: ` (Chapter ${chapterNum})`,
                                    size: sizes.small,
                                    italics: true,
                                    font: 'Times New Roman',
                                    color: colors.gray
                                })
                            ]
                        }));
                    });
                }
            });
        }
    });

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function buildAbstract(meta) {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'Abstract',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 240, line: 480 },
        children: [new TextRun({
            text: meta.abstract,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    if (meta.keywords) {
        elements.push(new Paragraph({
            spacing: { after: 240, line: 480 },
            indent: { firstLine: 720 },
            children: [
                new TextRun({
                    text: 'Keywords: ',
                    italics: true,
                    size: sizes.body,
                    font: 'Times New Roman'
                }),
                new TextRun({
                    text: meta.keywords.join(', '),
                    size: sizes.body,
                    font: 'Times New Roman'
                })
            ]
        }));
    }

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function parseHtml(html) {
    const elements = [];

    // Simple HTML parser - extract text from paragraphs and lists
    const paragraphs = html.split(/<\/?p>/gi).filter(p => p.trim());

    paragraphs.forEach(p => {
        // Strip remaining HTML tags for basic text
        const text = p.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
        if (text) {
            elements.push(new Paragraph({
                alignment: AlignmentType.JUSTIFIED,
                spacing: { after: 240, line: 480 },
                indent: { firstLine: 720 },
                children: [new TextRun({
                    text: text,
                    size: sizes.body,
                    font: 'Times New Roman'
                })]
            }));
        }
    });

    return elements;
}

async function buildAPAFigure(fig) {
    const elements = [];
    figureNumber++;

    elements.push(new Paragraph({ spacing: { before: 360 } }));

    // Try to load image
    const imgPath = fig.src.startsWith('figures/') ? fig.src : `figures/${fig.src}`;
    const fullPath = path.join('./', imgPath);

    if (fs.existsSync(fullPath)) {
        try {
            const imageData = fs.readFileSync(fullPath);
            elements.push(new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 240 },
                children: [
                    new ImageRun({
                        data: imageData,
                        transformation: { width: 450, height: 280 },
                        type: 'png'
                    })
                ]
            }));
        } catch (e) {
            console.warn(`Failed to load image: ${fullPath}`);
        }
    }

    // Figure caption
    elements.push(new Paragraph({
        spacing: { after: 120, line: 480 },
        children: [
            new TextRun({
                text: `Figure ${fig.number || figureNumber}`,
                italics: true,
                size: sizes.body,
                font: 'Times New Roman'
            })
        ]
    }));

    elements.push(new Paragraph({
        spacing: { after: 120, line: 480 },
        children: [
            new TextRun({
                text: fig.caption,
                italics: true,
                size: sizes.body,
                font: 'Times New Roman'
            })
        ]
    }));

    // Description
    if (fig.description) {
        elements.push(new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { after: 360, line: 480 },
            children: [
                new TextRun({
                    text: 'Note. ',
                    italics: true,
                    size: sizes.small,
                    font: 'Times New Roman'
                }),
                new TextRun({
                    text: fig.description,
                    size: sizes.small,
                    font: 'Times New Roman'
                })
            ]
        }));
    }

    return elements;
}

function buildAPATable(tableData) {
    const elements = [];
    tableNumber++;

    elements.push(new Paragraph({ spacing: { before: 360 } }));

    // Table number
    elements.push(new Paragraph({
        spacing: { after: 120, line: 480 },
        children: [new TextRun({
            text: `Table ${tableData.number || tableNumber}`,
            italics: true,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // Table title
    elements.push(new Paragraph({
        spacing: { after: 120, line: 480 },
        children: [new TextRun({
            text: tableData.caption,
            italics: true,
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // Build table rows
    const rows = [];

    if (tableData.headers) {
        rows.push(new TableRow({
            tableHeader: true,
            children: tableData.headers.map(header =>
                new TableCell({
                    children: [new Paragraph({
                        spacing: { after: 0 },
                        children: [new TextRun({
                            text: header,
                            bold: true,
                            size: sizes.body,
                            font: 'Times New Roman'
                        })]
                    })],
                    borders: {
                        top: { style: BorderStyle.SINGLE, size: 8, color: colors.black },
                        bottom: { style: BorderStyle.SINGLE, size: 4, color: colors.black },
                        left: { style: BorderStyle.NONE },
                        right: { style: BorderStyle.NONE }
                    }
                })
            )
        }));
    }

    if (tableData.rows) {
        tableData.rows.forEach((row, idx) => {
            const isLast = idx === tableData.rows.length - 1;
            rows.push(new TableRow({
                children: row.map(cell =>
                    new TableCell({
                        children: [new Paragraph({
                            spacing: { after: 0 },
                            children: [new TextRun({
                                text: String(cell),
                                size: sizes.body,
                                font: 'Times New Roman'
                            })]
                        })],
                        borders: {
                            top: { style: BorderStyle.NONE },
                            bottom: isLast
                                ? { style: BorderStyle.SINGLE, size: 8, color: colors.black }
                                : { style: BorderStyle.NONE },
                            left: { style: BorderStyle.NONE },
                            right: { style: BorderStyle.NONE }
                        }
                    })
                )
            }));
        });
    }

    elements.push(new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: rows
    }));

    // Note
    if (tableData.description) {
        elements.push(new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { before: 120, after: 360, line: 480 },
            children: [
                new TextRun({
                    text: 'Note. ',
                    italics: true,
                    size: sizes.small,
                    font: 'Times New Roman'
                }),
                new TextRun({
                    text: tableData.description,
                    size: sizes.small,
                    font: 'Times New Roman'
                })
            ]
        }));
    }

    return elements;
}

async function buildSection(section) {
    const elements = [];

    // Section heading
    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        alignment: AlignmentType.LEFT,
        spacing: { before: 360, after: 240, line: 480 },
        children: [new TextRun({
            text: `${section.section_key} ${section.title}`,
            bold: true,
            size: sizes.heading2,
            font: 'Times New Roman'
        })]
    }));

    // Content
    if (section.content_html) {
        elements.push(...parseHtml(section.content_html));
    }

    // Tables
    if (section.tables) {
        for (const table of section.tables) {
            elements.push(...buildAPATable(table));
        }
    }

    // Figures
    if (section.figures) {
        for (const fig of section.figures) {
            elements.push(...await buildAPAFigure(fig));
        }
    }

    // Subsections
    if (section.subsections) {
        for (const sub of section.subsections) {
            elements.push(...await buildSubsection(sub));
        }
    }

    return elements;
}

async function buildSubsection(subsection) {
    const elements = [];

    // Subsection heading
    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        alignment: AlignmentType.LEFT,
        spacing: { before: 240, after: 120, line: 480 },
        children: [new TextRun({
            text: `${subsection.section_key} ${subsection.title}`,
            bold: true,
            italics: true,
            size: sizes.heading3,
            font: 'Times New Roman'
        })]
    }));

    if (subsection.content_html) {
        elements.push(...parseHtml(subsection.content_html));
    }

    if (subsection.tables) {
        for (const table of subsection.tables) {
            elements.push(...buildAPATable(table));
        }
    }

    if (subsection.figures) {
        for (const fig of subsection.figures) {
            elements.push(...await buildAPAFigure(fig));
        }
    }

    return elements;
}

async function buildChapter(chapter) {
    const elements = [];

    // Chapter title
    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: `Chapter ${chapter.chapter_number}: ${chapter.title}`,
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    // Introduction
    if (chapter.introduction) {
        elements.push(...parseHtml(chapter.introduction));
    }

    // Sections
    if (chapter.sections) {
        for (const section of chapter.sections) {
            elements.push(...await buildSection(section));
        }
    }

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

function buildReferences(references) {
    const elements = [];

    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'References',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    references.forEach(ref => {
        let text = ref.citation;
        if (ref.url) text += ` ${ref.url}`;
        if (ref.doi) text += ` https://doi.org/${ref.doi}`;

        // APA hanging indent
        elements.push(new Paragraph({
            spacing: { after: 240, line: 480 },
            indent: { left: 720, hanging: 720 },
            children: [new TextRun({
                text: text,
                size: sizes.body,
                font: 'Times New Roman'
            })]
        }));
    });

    elements.push(new Paragraph({ children: [new PageBreak()] }));
    return elements;
}

async function buildAppendices() {
    const elements = [];

    // Appendices heading
    elements.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240, line: 480 },
        children: [new TextRun({
            text: 'Appendices',
            bold: true,
            size: sizes.heading1,
            font: 'Times New Roman'
        })]
    }));

    elements.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 360, line: 480 },
        indent: { firstLine: 720 },
        children: [new TextRun({
            text: 'The following appendices contain the complete Jupyter Notebook implementations for each machine learning model evaluated in this study.',
            size: sizes.body,
            font: 'Times New Roman'
        })]
    }));

    // Load manifest
    let manifest = null;
    const manifestPath = path.join(NOTEBOOK_IMAGES_PATH, 'manifest.json');
    if (fs.existsSync(manifestPath)) {
        manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
    }

    const notebooks = [
        { id: 'A', key: 'Ensemble', title: 'Ensemble Methods Implementation', subtitle: 'BaggingClassifier with Decision Tree Base Estimators' },
        { id: 'B', key: 'NonLinear', title: 'Non-Linear Methods Implementation', subtitle: 'DecisionTreeClassifier with Optimized Parameters' },
        { id: 'C', key: 'SupportVector', title: 'Linear Methods Implementation', subtitle: 'RidgeClassifier with Balanced Class Weights' }
    ];

    for (const nb of notebooks) {
        // Appendix heading
        elements.push(new Paragraph({
            heading: HeadingLevel.HEADING_2,
            alignment: AlignmentType.LEFT,
            spacing: { before: 480, after: 120, line: 480 },
            children: [new TextRun({
                text: `Appendix ${nb.id}: ${nb.title}`,
                bold: true,
                size: sizes.heading2,
                font: 'Times New Roman'
            })]
        }));

        elements.push(new Paragraph({
            spacing: { after: 240, line: 480 },
            children: [new TextRun({
                text: nb.subtitle,
                italics: true,
                size: sizes.body,
                font: 'Times New Roman'
            })]
        }));

        // Embed notebook images
        if (manifest && manifest[nb.key] && manifest[nb.key].pages) {
            const manifestData = manifest[nb.key];

            elements.push(new Paragraph({
                spacing: { after: 200, line: 480 },
                children: [
                    new TextRun({
                        text: 'Source File: ',
                        bold: true,
                        size: sizes.body,
                        font: 'Times New Roman'
                    }),
                    new TextRun({
                        text: manifestData.file,
                        size: sizes.body,
                        font: 'Consolas'
                    }),
                    new TextRun({
                        text: ` (${manifestData.pages.length} pages)`,
                        size: sizes.body,
                        italics: true,
                        font: 'Times New Roman'
                    })
                ]
            }));

            for (let i = 0; i < manifestData.pages.length; i++) {
                const imagePath = path.join(NOTEBOOK_IMAGES_PATH, manifestData.pages[i]);

                if (fs.existsSync(imagePath)) {
                    const imageData = fs.readFileSync(imagePath);

                    // Page header
                    elements.push(new Paragraph({
                        alignment: AlignmentType.RIGHT,
                        spacing: { before: 0, after: 60 },
                        children: [new TextRun({
                            text: `Page ${i + 1} of ${manifestData.pages.length}`,
                            size: 18,
                            italics: true,
                            color: colors.gray,
                            font: 'Times New Roman'
                        })]
                    }));

                    // Full-page notebook image
                    elements.push(new Paragraph({
                        alignment: AlignmentType.CENTER,
                        spacing: { before: 0, after: 0 },
                        children: [
                            new ImageRun({
                                data: imageData,
                                transformation: { width: 540, height: 765 },
                                type: 'png'
                            })
                        ]
                    }));

                    // Page break after each image
                    if (i < manifestData.pages.length - 1) {
                        elements.push(new Paragraph({ children: [new PageBreak()] }));
                    }
                }
            }
        }

        elements.push(new Paragraph({ children: [new PageBreak()] }));
    }

    return elements;
}

async function buildDocument(data) {
    const sections = [];
    const pageMargins = {
        top: 1440,    // 1 inch
        right: 1440,
        bottom: 1440,
        left: 1440
    };

    // Section 1: Title page (no header/footer)
    sections.push({
        properties: {
            page: { margin: pageMargins },
            titlePage: true
        },
        children: buildTitlePage(data.metadata)
    });

    // Section 2: Front matter (with header and footer)
    const frontMatter = [];
    frontMatter.push(...buildDeclaration(data.metadata));
    frontMatter.push(...buildAcknowledgments(data.metadata));
    frontMatter.push(...buildTableOfContents());
    frontMatter.push(...buildListOfFigures(data.chapters));
    frontMatter.push(...buildListOfTables(data.chapters));
    frontMatter.push(...buildAbstract(data.metadata));

    sections.push({
        properties: {
            page: { margin: pageMargins }
        },
        headers: {
            default: createHeader('DACS Assignment')
        },
        footers: {
            default: createFooter()
        },
        children: frontMatter
    });

    // Section 3: Main content
    const mainContent = [];
    for (const chapter of data.chapters) {
        if (chapter) {
            mainContent.push(...await buildChapter(chapter));
        }
    }
    if (data.references?.references) {
        mainContent.push(...buildReferences(data.references.references));
    }
    mainContent.push(...await buildAppendices());

    sections.push({
        properties: {
            page: { margin: pageMargins }
        },
        headers: {
            default: createHeader(SHORT_TITLE)
        },
        footers: {
            default: createFooter()
        },
        children: mainContent
    });

    return new Document({
        styles: {
            default: {
                document: {
                    run: {
                        font: 'Times New Roman',
                        size: sizes.body,
                        color: colors.black
                    },
                    paragraph: {
                        spacing: { line: 480, after: 0 },
                        alignment: AlignmentType.JUSTIFIED
                    }
                }
            }
        },
        numbering: {
            config: [
                {
                    reference: 'numbered-list',
                    levels: [{
                        level: 0,
                        format: LevelFormat.DECIMAL,
                        text: '%1.',
                        alignment: AlignmentType.LEFT,
                        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                    }]
                },
                {
                    reference: 'bullet-list',
                    levels: [{
                        level: 0,
                        format: LevelFormat.BULLET,
                        text: '•',
                        alignment: AlignmentType.LEFT,
                        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
                    }]
                }
            ]
        },
        sections: sections
    });
}

async function main() {
    console.log('Loading content...');
    const data = await loadContent();

    console.log('Building document...');
    const doc = await buildDocument(data);

    console.log('Saving DOCX...');
    const buffer = await Packer.toBuffer(doc);
    const outputPath = './output_test.docx';
    fs.writeFileSync(outputPath, buffer);

    console.log(`Document saved to: ${outputPath}`);
    console.log(`File size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);

    // Analyze document structure
    console.log('\n=== Document Analysis ===');
    console.log(`Chapters: ${data.chapters.filter(Boolean).length}`);

    let totalFigures = 0;
    let totalTables = 0;
    data.chapters.forEach(ch => {
        if (ch && ch.sections) {
            ch.sections.forEach(s => {
                if (s.figures) totalFigures += s.figures.length;
                if (s.tables) totalTables += s.tables.length;
                if (s.subsections) {
                    s.subsections.forEach(sub => {
                        if (sub.figures) totalFigures += sub.figures.length;
                        if (sub.tables) totalTables += sub.tables.length;
                    });
                }
            });
        }
    });

    console.log(`Figures: ${totalFigures}`);
    console.log(`Tables: ${totalTables}`);
    console.log(`References: ${data.references?.references?.length || 0}`);

    // Check notebook images
    if (fs.existsSync('./notebook-images/manifest.json')) {
        const manifest = JSON.parse(fs.readFileSync('./notebook-images/manifest.json', 'utf-8'));
        let totalPages = 0;
        Object.keys(manifest).forEach(key => {
            console.log(`Appendix ${key}: ${manifest[key].pages.length} pages`);
            totalPages += manifest[key].pages.length;
        });
        console.log(`Total notebook pages: ${totalPages}`);
    }
}

main().catch(console.error);
