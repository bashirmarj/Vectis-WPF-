import React from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

type AnimationType = 'fadeUp' | 'fadeLeft' | 'fadeRight' | 'scaleIn' | 'fadeIn';

interface AnimatedSectionProps {
  children: React.ReactNode;
  className?: string;
  animation?: AnimationType;
  delay?: number;
  duration?: number;
  threshold?: number;
}

const animationClasses: Record<AnimationType, { initial: string; visible: string }> = {
  fadeUp: {
    initial: 'opacity-0 translate-y-12',
    visible: 'opacity-100 translate-y-0',
  },
  fadeLeft: {
    initial: 'opacity-0 -translate-x-12',
    visible: 'opacity-100 translate-x-0',
  },
  fadeRight: {
    initial: 'opacity-0 translate-x-12',
    visible: 'opacity-100 translate-x-0',
  },
  scaleIn: {
    initial: 'opacity-0 scale-95',
    visible: 'opacity-100 scale-100',
  },
  fadeIn: {
    initial: 'opacity-0',
    visible: 'opacity-100',
  },
};

const AnimatedSection: React.FC<AnimatedSectionProps> = ({
  children,
  className,
  animation = 'fadeUp',
  delay = 0,
  duration = 700,
  threshold = 0.1,
}) => {
  const { ref, isVisible } = useScrollAnimation({ threshold });
  const { initial, visible } = animationClasses[animation];

  return (
    <div
      ref={ref}
      className={cn(
        'transition-all ease-out',
        isVisible ? visible : initial,
        className
      )}
      style={{
        transitionDuration: `${duration}ms`,
        transitionDelay: `${delay}ms`,
      }}
    >
      {children}
    </div>
  );
};

export default AnimatedSection;
