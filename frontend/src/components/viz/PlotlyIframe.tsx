import { useRef, useEffect, memo } from 'react';

interface PlotlyIframeProps {
  src: string;
  title?: string;
  className?: string;
  height?: string;
}

const PLOTLY_DARK_THEME_CSS = `
  body, .plotly, .js-plotly-plot, .plot-container {
    background: transparent !important;
    background-color: transparent !important;
  }
  .main-svg {
    background: transparent !important;
  }
  .bg {
    fill: transparent !important;
  }
  .gridlayer path {
    stroke: rgba(255,255,255,0.1) !important;
  }
  .zerolinelayer path {
    stroke: rgba(255,255,255,0.2) !important;
  }
  text, .xtick text, .ytick text, .gtitle, .g-xtitle text, .g-ytitle text {
    fill: #aaa !important;
  }
  .legendtext {
    fill: #bbb !important;
  }
  .modebar-btn path {
    fill: #888 !important;
  }
  .modebar-btn:hover path {
    fill: #00f2ea !important;
  }
  .hoverlayer .hovertext rect {
    fill: rgba(20,20,20,0.95) !important;
    stroke: #333 !important;
  }
  .hoverlayer .hovertext tspan {
    fill: #eee !important;
  }
`;

function PlotlyIframeBase({ src, title, className = '', height = '500px' }: PlotlyIframeProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const injectDarkTheme = () => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    try {
      const doc = iframe.contentDocument;
      if (!doc) return;

      const existingStyle = doc.getElementById('plotly-dark-theme');
      if (existingStyle) return;

      const style = doc.createElement('style');
      style.id = 'plotly-dark-theme';
      style.textContent = PLOTLY_DARK_THEME_CSS;
      doc.head.appendChild(style);
    } catch (e) {
      console.warn('Could not inject dark theme into iframe:', e);
    }
  };

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    iframe.addEventListener('load', injectDarkTheme);

    const retryTimeout = setTimeout(injectDarkTheme, 500);

    return () => {
      iframe.removeEventListener('load', injectDarkTheme);
      clearTimeout(retryTimeout);
    };
  }, [src]);

  return (
    <iframe
      ref={iframeRef}
      src={src}
      title={title || 'Interactive Chart'}
      className={`w-full border-0 ${className}`}
      style={{ height, background: 'transparent' }}
      sandbox="allow-scripts allow-same-origin"
    />
  );
}

export const PlotlyIframe = memo(PlotlyIframeBase);

export default PlotlyIframe;

