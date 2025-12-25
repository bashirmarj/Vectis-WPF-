import { useState, useEffect } from "react";
import { Mail, Phone, MapPin, Clock } from "lucide-react";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import AnimatedSection from "@/components/home/AnimatedSection";
import ParticleBackground from "@/components/home/ParticleBackground";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import capabilitiesImg from "@/assets/capabilities-bg.jpg";

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
    <div className="min-h-screen bg-[#F5F5F2]">
      <Navigation />
      <ParticleBackground />

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden z-10">
        <div
          className="absolute inset-0 opacity-20 bg-cover bg-center"
          style={{ backgroundImage: `url(${capabilitiesImg})` }}
        ></div>
        <div className="absolute inset-0 bg-gradient-to-b from-[#F5F5F2]/60 via-[#F5F5F2]/40 to-[#F5F5F2]"></div>
        <div className="container-custom relative z-10">
          <AnimatedSection animation="fadeUp">
            <div className="max-w-3xl">
              <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-snug tracking-normal">
                Contact Us
              </h1>
              <p className="text-xl text-gray-600 leading-relaxed">
                Get in touch to discuss your manufacturing project. Our team is ready to provide expert guidance and competitive quotes.
              </p>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Contact Form & Info */}
      <section className="relative z-10 border-t border-gray-200">
        <div className="container-custom section-spacing">
          <div className="grid lg:grid-cols-3 gap-12">
            {/* Contact Form */}
            <div className="lg:col-span-2">
              <AnimatedSection animation="fadeRight">
                <Card className="bg-white/80 backdrop-blur-md border border-gray-200 shadow-lg" id="send-message">
                  <CardContent className="p-8">
                    <div className="flex items-start gap-2 mb-6">
                      <div className="w-1 h-8 bg-primary rounded-full"></div>
                      <h2 className="text-3xl font-bold text-gray-900 tracking-tight font-mono uppercase">Send Us a Message</h2>
                    </div>
                    {isRateLimited && rateLimitRemaining && (
                      <div className="mb-6 p-4 bg-destructive/20 border border-destructive/30 rounded-lg">
                        <p className="text-destructive font-semibold">
                          ⏱️ Please wait {formatTimeRemaining(rateLimitRemaining)} before submitting another message.
                        </p>
                      </div>
                    )}
                    <form onSubmit={handleSubmit} className="space-y-6">
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <label htmlFor="name" className="block text-sm font-semibold mb-2 text-gray-700">
                            Name *
                          </label>
                          <Input
                            id="name"
                            name="name"
                            value={formData.name}
                            onChange={handleChange}
                            required
                            placeholder="John Doe"
                            className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400 focus:border-primary"
                          />
                        </div>
                        <div>
                          <label htmlFor="email" className="block text-sm font-semibold mb-2 text-gray-700">
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
                            className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400 focus:border-primary"
                          />
                        </div>
                      </div>

                      <div>
                        <label htmlFor="phone" className="block text-sm font-semibold mb-2 text-gray-700">
                          Phone
                        </label>
                        <Input
                          id="phone"
                          name="phone"
                          type="tel"
                          value={formData.phone}
                          onChange={handleChange}
                          placeholder="(123) 456-7890"
                          className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400 focus:border-primary"
                        />
                      </div>

                      <div>
                        <label htmlFor="message" className="block text-sm font-semibold mb-2 text-gray-700">
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
                          className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400 focus:border-primary resize-none"
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
              </AnimatedSection>
            </div>

            {/* Contact Information */}
            <div className="space-y-6">
              {contactInfo.map((info, index) => {
                const Icon = info.icon;
                return (
                  <AnimatedSection key={index} animation="fadeLeft" delay={index * 100}>
                    <Card className="bg-white/80 backdrop-blur-md border border-gray-200 hover:border-primary/50 transition-colors shadow-sm">
                      <CardContent className="p-6">
                        <div className="flex items-start gap-4">
                          <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center flex-shrink-0">
                            <Icon className="h-6 w-6 text-primary" />
                          </div>
                          <div>
                            <h3 className="font-bold text-lg mb-2 text-gray-900">{info.title}</h3>
                            {info.details.map((detail, idx) => (
                              <p key={idx} className="text-gray-600 text-sm">
                                {detail}
                              </p>
                            ))}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </AnimatedSection>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Contact;