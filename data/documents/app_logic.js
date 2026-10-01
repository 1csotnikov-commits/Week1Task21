// Пример модуля на JavaScript для структурного чанкинга.

function normalizeText(text) {
    return text.trim().toLowerCase();
}

function tokenize(text) {
    return text.split(/\s+/).filter(Boolean);
}

class TextProcessor {
    constructor(options) {
        this.options = options || {};
    }

    process(text) {
        const normalized = normalizeText(text);
        const tokens = tokenize(normalized);
        return { normalized, tokens };
    }
}

module.exports = { TextProcessor, normalizeText, tokenize };
