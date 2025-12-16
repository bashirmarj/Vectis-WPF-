import React, { useEffect, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { MapPin, Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

interface AddressAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

declare global {
  interface Window {
    google: any;
    initGooglePlaces: () => void;
  }
}

export const AddressAutocomplete: React.FC<AddressAutocompleteProps> = ({
  value,
  onChange,
  placeholder = "Start typing for suggestions...",
  className,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const loadGooglePlaces = async () => {
      // Check if already loaded
      if (window.google?.maps?.places) {
        initAutocomplete();
        setIsLoading(false);
        return;
      }

      try {
        // Fetch API key from edge function
        const { data, error: fetchError } = await supabase.functions.invoke('get-maps-api-key');
        
        if (fetchError || !data?.apiKey) {
          throw new Error('Failed to load address autocomplete');
        }

        // Load Google Maps script
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${data.apiKey}&libraries=places`;
        script.async = true;
        script.defer = true;
        
        script.onload = () => {
          if (isMounted) {
            initAutocomplete();
            setIsLoading(false);
          }
        };
        
        script.onerror = () => {
          if (isMounted) {
            setError('Failed to load address suggestions');
            setIsLoading(false);
          }
        };

        document.head.appendChild(script);
      } catch (err) {
        if (isMounted) {
          console.error('Google Places error:', err);
          setError('Address suggestions unavailable');
          setIsLoading(false);
        }
      }
    };

    const initAutocomplete = () => {
      if (!inputRef.current || !window.google?.maps?.places) return;

      autocompleteRef.current = new window.google.maps.places.Autocomplete(
        inputRef.current,
        {
          types: ['address'],
          fields: ['formatted_address', 'address_components'],
        }
      );

      autocompleteRef.current.addListener('place_changed', () => {
        const place = autocompleteRef.current.getPlace();
        if (place?.formatted_address) {
          onChange(place.formatted_address);
        }
      });
    };

    loadGooglePlaces();

    return () => {
      isMounted = false;
      if (autocompleteRef.current) {
        window.google?.maps?.event?.clearInstanceListeners(autocompleteRef.current);
      }
    };
  }, [onChange]);

  return (
    <div className="relative">
      <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <MapPin className="w-4 h-4" />
        )}
      </div>
      <Input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={error || placeholder}
        className={`pl-10 h-9 ${className || ''}`}
        disabled={isLoading}
      />
    </div>
  );
};
