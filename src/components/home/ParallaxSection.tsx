import React from 'react';
import { useParallax } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

interface ParallaxSectionProps {
  children: React.ReactNode;
  backgroundImage?: string;
  speed?: number;
  overlay?: boolean;
  overlayOpacity?: number;
  className?: string;
  contentClassName?: string;
}

const ParallaxSection: React.FC<ParallaxSectionProps> = ({
  children,
  backgroundImage,
  speed = 0.3,
  overlay = true,
  overlayOpacity = 0.7,
  className,
  contentClassName,
}) => {
  const { ref, offset } = useParallax(speed);

  return (
    <div ref={ref} className={cn('relative overflow-hidden', className)}>
      {backgroundImage && (
        <div
          className="absolute inset-0 bg-cover bg-center will-change-transform"
          style={{
            backgroundImage: `url(${backgroundImage})`,
            transform: `translateY(${offset}px) scale(1.1)`,
          }}
        />
      )}
      {overlay && (
        <div
          className="absolute inset-0 bg-accent"
          style={{ opacity: overlayOpacity }}
        />
      )}
      <div className={cn('relative z-10', contentClassName)}>{children}</div>
    </div>
  );
};

export default ParallaxSection;
