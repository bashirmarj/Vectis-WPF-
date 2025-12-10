import React, { useEffect, useState } from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

interface AnimatedCounterProps {
  value: string;
  label: string;
  duration?: number;
  delay?: number;
  className?: string;
}

const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  value,
  label,
  duration = 2000,
  delay = 0,
  className,
}) => {
  const { ref, isVisible } = useScrollAnimation({ threshold: 0.3 });
  const [displayValue, setDisplayValue] = useState('0');

  useEffect(() => {
    if (!isVisible) return;

    const numericMatch = value.match(/^(\d+)/);
    if (!numericMatch) {
      setDisplayValue(value);
      return;
    }

    const targetNumber = parseInt(numericMatch[1], 10);
    const suffix = value.replace(/^\d+/, '');
    const startTime = Date.now() + delay;
    const endTime = startTime + duration;

    const animate = () => {
      const now = Date.now();
      
      if (now < startTime) {
        requestAnimationFrame(animate);
        return;
      }

      if (now >= endTime) {
        setDisplayValue(value);
        return;
      }

      const progress = (now - startTime) / duration;
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentValue = Math.floor(targetNumber * easeOutQuart);
      setDisplayValue(`${currentValue}${suffix}`);
      requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [isVisible, value, duration, delay]);

  return (
    <div
      ref={ref}
      className={cn(
        'text-center transition-all duration-700',
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4',
        className
      )}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className="text-3xl md:text-4xl lg:text-5xl font-bold text-primary mb-2 tabular-nums">
        {displayValue}
      </div>
      <div className="text-xs md:text-sm text-muted-foreground uppercase tracking-widest">
        {label}
      </div>
    </div>
  );
};

export default AnimatedCounter;
