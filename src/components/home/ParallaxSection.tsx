import React from 'react';
import { cn } from '@/lib/utils';

interface ParallaxSectionProps {
  children: React.ReactNode;
  backgroundImage?: string;
  overlay?: boolean;
  overlayOpacity?: number;
  className?: string;
  contentClassName?: string;
}

const ParallaxSection: React.FC<ParallaxSectionProps> = ({
  children,
  backgroundImage,
  overlay = true,
  overlayOpacity = 0.7,
  className,
  contentClassName,
}) => {
  return (
    <div className={cn('relative overflow-hidden', className)}>
      {backgroundImage && (
        <div
          className="absolute inset-x-0 bg-cover bg-center"
          style={{
            backgroundImage: `url(${backgroundImage})`,
            height: '150%',
            top: '-25%',
          }}
        />
      )}
      {overlay && (
        <div
          className="absolute inset-0 bg-black"
          style={{ opacity: overlayOpacity }}
        />
      )}
      <div className={cn('relative z-10', contentClassName)}>{children}</div>
    </div>
  );
};

export default ParallaxSection;
