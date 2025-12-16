import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CapabilityCardProps {
  title: string;
  description: string;
  image: string;
  link: string;
  className?: string;
  isVisible?: boolean;
  delay?: number;
}

const CapabilityCard: React.FC<CapabilityCardProps> = ({
  title,
  description,
  image,
  link,
  className,
  isVisible = true,
  delay = 0,
}) => {
  return (
    <Link
      to={link}
      className={cn(
        'group relative overflow-hidden rounded-lg bg-card border border-border transition-all duration-500',
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8',
        className
      )}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {/* Image */}
      <div className="relative h-56 overflow-hidden">
        <img
          src={image}
          alt={title}
          className="w-full h-full object-cover grayscale transition-all duration-700 group-hover:scale-110 group-hover:grayscale-0"
        />
      </div>

      {/* Content */}
      <div className="p-6">
        {/* Capability label with arrow */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-primary">
            Capability
          </span>
          <ArrowRight className="w-4 h-4 text-primary transition-transform duration-300 group-hover:translate-x-1" />
        </div>
        <h3 className="text-xl font-bold mb-2 uppercase group-hover:text-primary transition-colors duration-300">
          {title}
        </h3>
        <p className="text-muted-foreground text-sm leading-relaxed">
          {description}
        </p>
      </div>

      {/* Bottom accent line */}
      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left" />
    </Link>
  );
};

export default CapabilityCard;
