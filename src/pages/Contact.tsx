import { useState, useEffect } from "react";
import { Mail, Phone, MapPin, Clock } from "lucide-react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    phone: "",
    message: "",
  });
  const [loading, setLoading] = useState(false);
  const [rateLimitRemaining, setRateLimitRemaining] = useState<number | null>(null);
  const [isRateLimited, setIsRateLimited] = useState(false);

  useEffect(() => {
    // Scroll to send-message section if hash is present
    if (window.location.hash === '#send-message') {
      setTimeout(() => {
        const element = document.getElementById('send-message');
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    }
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (rateLimitRemaining !== null && rateLimitRemaining > 0) {
      interval = setInterval(() => {
        setRateLimitRemaining((prev) => {
          if (prev === null || prev <= 1) {
            setIsRateLimited(false);
            return null;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [rateLimitRemaining]);

  const formatTimeRemaining = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs}s`;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const { error } = await supabase.functions.invoke('send-contact-message', {
        body: formData,
      });

      if (error) {
        if (error.message.includes('rate_limit_exceeded')) {
          const errorData = JSON.parse(error.message);
          setRateLimitRemaining(errorData.remainingSeconds);
          setIsRateLimited(true);
          toast.error(
            `Please wait ${formatTimeRemaining(errorData.remainingSeconds)} before submitting another message.`,
            { duration: 5000 }
          );
        } else {
          toast.error("Failed to send message. Please try again.");
        }
      } else {
        toast.success("Thank you for your message! We'll get back to you within 24 hours.");
        setFormData({
          name: "",
          email: "",
          phone: "",
          message: "",
        });
      }
    } catch (error: any) {
      console.error('Contact form error:', error);
      toast.error("An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const contactInfo = [
    {
      icon: Phone,
      title: "Phone",
      details: ["Main: (123) 456-7890", "Toll Free: (800) 123-4567"],
    },
    {
      icon: Mail,
      title: "Email",
      details: ["General: info@vectis-mfg.com", "Quotes: quotes@vectis-mfg.com"],
    },
    {
      icon: MapPin,
      title: "Address",
      details: ["123 Industrial Way", "Manufacturing City, ST 12345"],
    },
    {
      icon: Clock,
      title: "Business Hours",
      details: ["Monday - Friday: 7:00 AM - 6:00 PM", "Saturday: 8:00 AM - 2:00 PM"],
    },
  ];

  return (
    <div className="min-h-screen">
      <Navigation />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 bg-accent text-accent-foreground">
        <div className="container-custom">
          <div className="max-w-3xl">
            <h1 className="text-white mb-6">Contact Us</h1>
            <p className="text-xl text-gray-200 leading-relaxed">
              Get in touch to discuss your manufacturing project. Our team is ready to provide expert guidance and competitive quotes.
            </p>
          </div>
        </div>
      </section>

      {/* Contact Form & Info */}
      <section className="section-spacing bg-background">
        <div className="container-custom">
          <div className="grid lg:grid-cols-3 gap-12">
            {/* Contact Form */}
            <div className="lg:col-span-2">
              <Card className="border-2" id="send-message">
                <CardContent className="p-8">
                  <h2 className="text-3xl font-bold mb-6">Send Us a Message</h2>
                  {isRateLimited && rateLimitRemaining && (
                    <div className="mb-6 p-4 bg-destructive/10 border border-destructive rounded-lg">
                      <p className="text-destructive font-semibold">
                        ⏱️ Please wait {formatTimeRemaining(rateLimitRemaining)} before submitting another message.
                      </p>
                    </div>
                  )}
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <label htmlFor="name" className="block text-sm font-semibold mb-2">
                          Name *
                        </label>
                        <Input
                          id="name"
                          name="name"
                          value={formData.name}
                          onChange={handleChange}
                          required
                          placeholder="John Doe"
                        />
                      </div>
                      <div>
                        <label htmlFor="email" className="block text-sm font-semibold mb-2">
                          Email *
                        </label>
                        <Input
                          id="email"
                          name="email"
                          type="email"
                          value={formData.email}
                          onChange={handleChange}
                          required
                          placeholder="john@company.com"
                        />
                      </div>
                    </div>

                    <div>
                      <label htmlFor="phone" className="block text-sm font-semibold mb-2">
                        Phone
                      </label>
                      <Input
                        id="phone"
                        name="phone"
                        type="tel"
                        value={formData.phone}
                        onChange={handleChange}
                        placeholder="(123) 456-7890"
                      />
                    </div>

                    <div>
                      <label htmlFor="message" className="block text-sm font-semibold mb-2">
                        Message *
                      </label>
                      <Textarea
                        id="message"
                        name="message"
                        value={formData.message}
                        onChange={handleChange}
                        required
                        rows={6}
                        placeholder="Please provide details about your inquiry..."
                      />
                    </div>

                    <Button 
                      type="submit" 
                      size="lg" 
                      className="w-full md:w-auto"
                      disabled={loading || isRateLimited}
                    >
                      {loading ? 'Sending...' : isRateLimited && rateLimitRemaining ? `Please wait ${formatTimeRemaining(rateLimitRemaining)}` : 'Send Message'}
                    </Button>
                  </form>
                </CardContent>
              </Card>
            </div>

            {/* Contact Information */}
            <div className="space-y-6">
              {contactInfo.map((info, index) => {
                const Icon = info.icon;
                return (
                  <Card key={index} className="border-2">
                    <CardContent className="p-6">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
                          <Icon className="h-6 w-6 text-primary-foreground" />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg mb-2">{info.title}</h3>
                          {info.details.map((detail, idx) => (
                            <p key={idx} className="text-muted-foreground text-sm">
                              {detail}
                            </p>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}

              <Card className="border-2 bg-accent text-accent-foreground">
                <CardContent className="p-6">
                  <h3 className="font-bold text-lg mb-2 text-white">Emergency Services</h3>
                  <p className="text-gray-200 text-sm mb-2">
                    Need urgent manufacturing support?
                  </p>
                  <p className="text-gray-200 text-sm">
                    24/7 Emergency: <span className="font-bold text-primary">(800) 911-MACH</span>
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Contact;
