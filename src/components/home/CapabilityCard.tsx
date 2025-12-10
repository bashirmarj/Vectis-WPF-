import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CapabilityCardProps {
  title: string;
  description: string;
  image: string;
  icon: LucideIcon;
  link: string;
  className?: string;
  isVisible?: boolean;
  delay?: number;
}

const CapabilityCard: React.FC<CapabilityCardProps> = ({
  title,
  description,
  image,
  icon: Icon,
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
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-accent via-accent/50 to-transparent opacity-80 group-hover:opacity-90 transition-opacity duration-300" />
        
        {/* Icon badge */}
        <div className="absolute bottom-4 left-4 w-12 h-12 bg-primary rounded-lg flex items-center justify-center transform transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3">
          <Icon className="w-6 h-6 text-primary-foreground" />
        </div>

        {/* Arrow indicator */}
        <div className="absolute bottom-4 right-4 w-10 h-10 bg-white/10 backdrop-blur-sm rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-x-4 group-hover:translate-x-0">
          <ArrowRight className="w-5 h-5 text-white" />
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <h3 className="text-xl font-bold mb-2 group-hover:text-primary transition-colors duration-300">
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
