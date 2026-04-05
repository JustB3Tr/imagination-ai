'use client';

import { memo, useEffect, useState } from 'react';

interface MathRendererProps {
  content: string;
}

interface KaTeXModule {
  renderToString: (latex: string, options: { displayMode: boolean; throwOnError: boolean; output: string; strict: boolean }) => string;
}

// Global KaTeX instance
let katexModule: KaTeXModule | null = null;
let katexLoading: Promise<KaTeXModule> | null = null;

// Load KaTeX from CDN
function loadKaTeX(): Promise<KaTeXModule> {
  if (katexModule) return Promise.resolve(katexModule);
  if (katexLoading) return katexLoading;

  katexLoading = new Promise((resolve, reject) => {
    // Load CSS first
    if (!document.querySelector('link[href*="katex"]')) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
      document.head.appendChild(link);
    }

    // Load JS
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
    script.onload = () => {
      katexModule = (window as unknown as { katex: KaTeXModule }).katex;
      resolve(katexModule);
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });

  return katexLoading;
}

// Render LaTeX to HTML string using KaTeX
function renderMath(katex: KaTeXModule, latex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(latex, {
      displayMode,
      throwOnError: false,
      output: 'html',
      strict: false,
    });
  } catch {
    return `<span class="text-red-400">${latex}</span>`;
  }
}

// Parse and render content with math
function processContent(katex: KaTeXModule, text: string): string {
  let result = text;
  
  // First, handle display math ($$...$$) - these should be on their own line
  result = result.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
    const rendered = renderMath(katex, math.trim(), true);
    return `<div class="my-4 overflow-x-auto flex justify-center">${rendered}</div>`;
  });
  
  // Then handle inline math ($...$)
  result = result.replace(/\$([^\$\n]+?)\$/g, (_, math) => {
    const rendered = renderMath(katex, math.trim(), false);
    return rendered;
  });
  
  // Format markdown-like syntax
  // Bold
  result = result.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
  // Italic (not preceded or followed by *)
  result = result.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  // Inline code
  result = result.replace(/`([^`]+)`/g, '<code class="bg-secondary/50 px-1.5 py-0.5 rounded text-sm font-mono text-accent">$1</code>');
  // Line breaks
  result = result.replace(/\n/g, '<br/>');
  
  return result;
}

export const MathRenderer = memo(function MathRenderer({ content }: MathRendererProps) {
  const [renderedContent, setRenderedContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    loadKaTeX().then((katex) => {
      if (mounted) {
        const processed = processContent(katex, content);
        setRenderedContent(processed);
        setIsLoading(false);
      }
    }).catch(() => {
      if (mounted) {
        // Fallback: just show content with basic formatting
        let result = content;
        result = result.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        result = result.replace(/\n/g, '<br/>');
        setRenderedContent(result);
        setIsLoading(false);
      }
    });

    return () => { mounted = false; };
  }, [content]);

  if (isLoading) {
    return (
      <div className="prose prose-invert max-w-none text-foreground leading-relaxed animate-pulse">
        <div className="h-4 bg-secondary/30 rounded w-3/4 mb-2"></div>
        <div className="h-4 bg-secondary/30 rounded w-1/2"></div>
      </div>
    );
  }

  return (
    <div 
      className="prose prose-invert max-w-none text-foreground leading-relaxed [&_.katex]:text-foreground [&_.katex-display]:my-4 [&_.katex-display]:overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: renderedContent || '' }}
    />
  );
});
