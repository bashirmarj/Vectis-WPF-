import { useState, useEffect, useRef, useCallback } from 'react';
import { Input } from '@/components/ui/input';
import { MapPin, Loader2 } from 'lucide-react';

// Extend Window type for Google Maps
declare global {
  interface Window {
    google?: {
      maps?: {
        places?: {
          AutocompleteService: new () => GoogleAutocompleteService;
          PlacesServiceStatus: {
            OK: string;
          };
        };
      };
    };
  }
}

interface GoogleAutocompleteService {
  getPlacePredictions: (
    request: { input: string; types: string[] },
    callback: (predictions: AutocompletePrediction[] | null, status: string) => void
  ) => void;
}

interface AutocompletePrediction {
  place_id: string;
  description: string;
  structured_formatting: {
    main_text: string;
    secondary_text: string;
  };
}

interface AddressAutocompleteProps {
  value: string;
  onChange: (address: string) => void;
  placeholder?: string;
  className?: string;
  required?: boolean;
}

// Load Google Maps API script dynamically
const loadGoogleMapsScript = (apiKey: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    if (window.google?.maps?.places) {
      resolve();
      return;
    }

    // Check if script already exists
    const existingScript = document.querySelector('script[src*="maps.googleapis.com"]');
    if (existingScript) {
      existingScript.addEventListener('load', () => resolve());
      return;
    }

    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places`;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Google Maps API'));
    document.head.appendChild(script);
  });
};

export function AddressAutocomplete({
  value,
  onChange,
  placeholder = "Start typing an address...",
  className = "",
  required = false,
}: AddressAutocompleteProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<AutocompletePrediction[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const autocompleteServiceRef = useRef<GoogleAutocompleteService | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize Google Maps API
  useEffect(() => {
    const initGoogleMaps = async () => {
      try {
        // Get API key from edge function or environment
        const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
        
        if (!apiKey) {
          // Fallback to basic input without autocomplete
          setError('Address autocomplete unavailable');
          setIsLoading(false);
          return;
        }

        await loadGoogleMapsScript(apiKey);
        if (window.google?.maps?.places) {
          autocompleteServiceRef.current = new window.google.maps.places.AutocompleteService();
        }
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to load Google Maps:', err);
        setError('Address autocomplete unavailable');
        setIsLoading(false);
      }
    };

    initGoogleMaps();
  }, []);

  // Handle input changes and fetch suggestions
  const handleInputChange = useCallback((inputValue: string) => {
    onChange(inputValue);

    if (!autocompleteServiceRef.current || inputValue.length < 3) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    autocompleteServiceRef.current.getPlacePredictions(
      {
        input: inputValue,
        types: ['address'],
      },
      (predictions, status) => {
        if (status === 'OK' && predictions) {
          setSuggestions(predictions);
          setShowSuggestions(true);
        } else {
          setSuggestions([]);
          setShowSuggestions(false);
        }
      }
    );
  }, [onChange]);

  // Handle suggestion selection
  const handleSelectSuggestion = useCallback((suggestion: AutocompletePrediction) => {
    onChange(suggestion.description);
    setSuggestions([]);
    setShowSuggestions(false);
    inputRef.current?.blur();
  }, [onChange]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle keyboard navigation
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!showSuggestions || suggestions.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => (prev < suggestions.length - 1 ? prev + 1 : prev));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => (prev > 0 ? prev - 1 : -1));
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
          handleSelectSuggestion(suggestions[selectedIndex]);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
    }
  }, [showSuggestions, suggestions, selectedIndex, handleSelectSuggestion]);

  return (
    <div ref={containerRef} className="relative">
      <div className="relative">
        <Input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => handleInputChange(e.target.value)}
          onFocus={() => value.length >= 3 && suggestions.length > 0 && setShowSuggestions(true)}
          onKeyDown={handleKeyDown}
          placeholder={isLoading ? "Loading..." : placeholder}
          className={`pl-10 ${className}`}
          disabled={isLoading}
          required={required}
        />
        <div className="absolute left-3 top-1/2 -translate-y-1/2">
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin text-white/40" />
          ) : (
            <MapPin className="h-4 w-4 text-white/40" />
          )}
        </div>
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute z-50 w-full mt-1 bg-black/95 border border-white/20 rounded-md shadow-lg max-h-60 overflow-auto">
          {suggestions.map((suggestion, index) => (
            <button
              key={suggestion.place_id}
              type="button"
              className={`w-full px-4 py-3 text-left text-sm transition-colors ${
                index === selectedIndex
                  ? 'bg-primary/20 text-white'
                  : 'text-white/80 hover:bg-white/10'
              }`}
              onClick={() => handleSelectSuggestion(suggestion)}
              onMouseEnter={() => setSelectedIndex(index)}
            >
              <div className="flex items-start gap-2">
                <MapPin className="h-4 w-4 mt-0.5 text-primary flex-shrink-0" />
                <div>
                  <div className="font-medium">{suggestion.structured_formatting.main_text}</div>
                  <div className="text-xs text-white/60">{suggestion.structured_formatting.secondary_text}</div>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Error fallback message */}
      {error && (
        <p className="text-xs text-white/50 mt-1">{error} - Enter address manually</p>
      )}
    </div>
  );
}
