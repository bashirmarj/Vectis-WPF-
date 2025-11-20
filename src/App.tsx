import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Index from "./pages/Index";
import About from "./pages/About";
import Services from "./pages/Services";
import PrototypeDesign from "./pages/services/PrototypeDesign";
import CustomParts from "./pages/services/CustomParts";
import PrototypingServices from "./pages/services/PrototypingServices";
import TurnkeySolutions from "./pages/services/TurnkeySolutions";
import Capabilities from "./pages/Capabilities";
import Projects from "./pages/Projects";
import Contact from "./pages/Contact";
import Auth from "./pages/Auth";
import Admin from "./pages/Admin";
import QuotationDetails from "./pages/admin/QuotationDetails";
import PricingSettings from "./pages/admin/PricingSettings";
import AdminSetup from "./pages/admin/AdminSetup";
import Validation from "./pages/Validation";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/about" element={<About />} />
            <Route path="/services" element={<Services />} />
            <Route path="/services/prototype-design" element={<PrototypeDesign />} />
            <Route path="/services/custom-parts" element={<CustomParts />} />
            <Route path="/services/prototyping" element={<PrototypingServices />} />
            <Route path="/services/turnkey-solutions" element={<TurnkeySolutions />} />
            <Route path="/capabilities" element={<Capabilities />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/auth" element={<Auth />} />
            <Route path="/admin" element={<Admin />} />
            <Route path="/admin/quotations/:id" element={<QuotationDetails />} />
            <Route path="/admin/pricing-settings" element={<PricingSettings />} />
            <Route path="/admin/setup" element={<AdminSetup />} />
            <Route path="/validation" element={<Validation />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
