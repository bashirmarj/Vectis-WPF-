import React, { useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ScrollIndicatorProps {
  className?: string;
  hideAfterScroll?: number;
}

const ScrollIndicator: React.FC<ScrollIndicatorProps> = ({
  className,
  hideAfterScroll = 100,
}) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      setIsVisible(window.scrollY < hideAfterScroll);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [hideAfterScroll]);

  const scrollToContent = () => {
    window.scrollTo({
      top: window.innerHeight - 80,
      behavior: 'smooth',
    });
  };

  return (
    <button
      onClick={scrollToContent}
      className={cn(
        'flex flex-col items-center gap-2 transition-all duration-500 cursor-pointer group',
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none',
        className
      )}
      aria-label="Scroll to content"
    >
      <span className="text-xs uppercase tracking-widest text-white/70 group-hover:text-white transition-colors">
        Scroll
      </span>
      <div className="relative">
        <ChevronDown className="w-6 h-6 text-white/70 group-hover:text-white animate-bounce" />
      </div>
    </button>
  );
};

export default ScrollIndicator;
