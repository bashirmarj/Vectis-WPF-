import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Menu, X, Phone, ChevronDown, User, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { NavigationMenu, NavigationMenuContent, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, NavigationMenuTrigger } from "@/components/ui/navigation-menu";
import { useAuth } from "@/contexts/AuthContext";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import logo from "@/assets/logo.png";
import { supabase } from "@/integrations/supabase/client";

const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { user, signOut, isAdmin, checkingRole } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleSignOut = async () => {
    await signOut();
    navigate('/');
  };

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);
  const navItems = [{
    name: "Home",
    path: "/"
  }, {
    name: "About",
    path: "/about",
    subItems: [{
      name: "Our Story",
      path: "/about#story"
    }, {
      name: "Team",
      path: "/about#team"
    }, {
      name: "Quality Standards",
      path: "/about#quality"
    }]
  }, {
    name: "Services",
    path: "/services"
  }, {
    name: "Contact",
    path: "/contact"
  }];
  const isActive = (path: string) => location.pathname === path;
  return <nav className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-md border-b border-white/10">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <div className="bg-white rounded-lg px-2 py-1">
              <img src={logo} alt="Vectis Manufacturing Logo" className="h-12 w-auto" />
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-bold text-primary-foreground leading-tight">VECTIS</span>
              <span className="text-sm text-primary-foreground/80 leading-tight">Manufacturing</span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <NavigationMenu className="hidden md:flex">
            <NavigationMenuList>
              {navItems.map(item => <NavigationMenuItem key={item.path}>
                  {item.subItems ? <>
                      <NavigationMenuTrigger className={`!bg-transparent ${isActive(item.path) ? "text-primary !bg-primary/10" : "text-accent-foreground hover:text-primary hover:!bg-primary/5"}`}>
                        {item.name}
                      </NavigationMenuTrigger>
                      <NavigationMenuContent>
                        <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px] bg-neutral-950/95 backdrop-blur-md border border-white/10 shadow-lg z-50">
                          {item.subItems.map(subItem => <li key={subItem.path}>
                              <NavigationMenuLink asChild>
                                <Link to={subItem.path} className="block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors text-accent-foreground hover:bg-primary/10 hover:text-primary focus:bg-primary/10 focus:text-primary">
                                  <div className="text-sm font-medium leading-none">{subItem.name}</div>
                                </Link>
                              </NavigationMenuLink>
                            </li>)}
                        </ul>
                      </NavigationMenuContent>
                    </> : <Link to={item.path} className={`inline-flex h-9 w-max items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors hover:bg-primary/5 hover:text-primary focus:bg-primary/5 focus:text-primary focus:outline-none disabled:pointer-events-none disabled:opacity-50 ${isActive(item.path) ? "text-primary bg-primary/10" : "text-primary-foreground"}`}>
                      {item.name}
                    </Link>}
                </NavigationMenuItem>)}
            </NavigationMenuList>
          </NavigationMenu>

          {/* CTA Buttons */}
          <div className="hidden md:flex items-center gap-3">
            <Button asChild>
              <Link to="/contact">Get Quote</Link>
            </Button>
            {user ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <User className="h-5 w-5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {!checkingRole && isAdmin && (
                    <>
                      <DropdownMenuItem onClick={() => navigate('/admin')}>
                        Admin Dashboard
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                    </>
                  )}
                  <DropdownMenuItem onClick={handleSignOut}>
                    <LogOut className="mr-2 h-4 w-4" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button variant="ghost" onClick={() => navigate('/auth')}>
                Sign In
              </Button>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button className="md:hidden text-primary-foreground p-2" onClick={() => setIsOpen(!isOpen)} aria-label="Toggle menu">
            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && <div className="md:hidden border-t border-white/10 bg-black/90 backdrop-blur-md">
            <ScrollArea className="h-[calc(100vh-5rem)]">
              <div className="flex flex-col space-y-2 py-4">
                {navItems.map(item => <div key={item.path}>
                    <Link to={item.path} onClick={() => setIsOpen(false)} className={`flex items-center justify-between px-4 py-3 rounded-md text-sm font-semibold transition-colors ${isActive(item.path) ? "text-primary bg-primary/10" : "text-primary-foreground hover:text-primary hover:bg-primary/5"}`}>
                      {item.name}
                    </Link>
                    {item.subItems && <div className="ml-4 mt-2 space-y-2">
                        {item.subItems.map(subItem => <Link key={subItem.path} to={subItem.path} onClick={() => setIsOpen(false)} className="block px-4 py-2 text-sm text-muted-foreground hover:text-primary transition-colors">
                            {subItem.name}
                          </Link>)}
                      </div>}
                  </div>)}
                <div className="pt-4 px-4 space-y-3">
                  <Button asChild className="w-full" onClick={() => setIsOpen(false)}>
                    <Link to="/contact">Get Quote</Link>
                  </Button>
                  {user ? (
                    <>
                      {!checkingRole && isAdmin && (
                        <Button 
                          variant="outline" 
                          className="w-full" 
                          onClick={() => {
                            setIsOpen(false);
                            navigate('/admin');
                          }}
                        >
                          <User className="mr-2 h-4 w-4" />
                          Admin Dashboard
                        </Button>
                      )}
                      <Button 
                        variant="ghost" 
                        className="w-full" 
                        onClick={() => {
                          setIsOpen(false);
                          handleSignOut();
                        }}
                      >
                        <LogOut className="mr-2 h-4 w-4" />
                        Sign Out
                      </Button>
                    </>
                  ) : (
                    <Button 
                      variant="outline" 
                      className="w-full" 
                      onClick={() => {
                        setIsOpen(false);
                        navigate('/auth');
                      }}
                    >
                      Sign In
                    </Button>
                  )}
                </div>
              </div>
            </ScrollArea>
          </div>}
      </div>
    </nav>;
};
export default Navigation;