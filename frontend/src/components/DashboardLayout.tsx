import { ReactNode, useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Video, 
  ScissorsLineDashed, 
  EyeOff, 
  Minimize2, 
  History, 
  LogOut,
  Menu,
  X,
  User
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface DashboardLayoutProps {
  children: ReactNode;
}

const navItems = [
  { icon: ScissorsLineDashed, label: 'Summarization', path: '/summarization' },
  { icon: EyeOff, label: 'Blurring', path: '/blurring' },
  { icon: Minimize2, label: 'Compression', path: '/compression' },
  { icon: History, label: 'History', path: '/history' },
  { icon: User, label: 'Profile', path: '/profile' },
];

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const { signOut, user } = useAuth();
  const navigate = useNavigate();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleSignOut = async () => {
    await signOut();
    navigate('/auth');
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Desktop Navbar */}
      <header className="sticky top-0 z-50 w-full border-b border-border/50 gradient-card">
        <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
          {/* Logo */}
          <div className="flex items-center cursor-pointer" onClick={() => navigate('/dashboard')}>
            {/* <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white shadow-glow"> */}
              {/* <Video className="h-4 w-4 text-primary-foreground" /> */}
              <img src="/logo-n-1080.png" alt="NeuroClip" className="h-24 w-24 scale-110 object-contain" />
            {/* </div> */}
            <span className="text-lg font-bold hidden sm:inline">NeuroClip</span>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all hover:bg-muted',
                    isActive && 'bg-muted text-primary'
                  )
                }
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </NavLink>
            ))}
          </nav>

          {/* Right Side Actions */}
          <div className="flex items-center gap-2">
            <ThemeToggle />
            
            {/* Desktop User Menu */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild className="hidden lg:flex">
                <Button variant="ghost" size="icon">
                  <User className="h-5 w-5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">Account</p>
                    <p className="text-xs leading-none text-muted-foreground truncate max-w-[200px]">
                      {user?.email}
                    </p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut}>
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Mobile/Tablet Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={() => setDrawerOpen(!drawerOpen)}
            >
              <Menu className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto p-4 md:p-6 overflow-auto">
        {children}
      </main>

      {/* Mobile/Tablet Right Drawer */}
      <AnimatePresence>
        {drawerOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden"
              onClick={() => setDrawerOpen(false)}
            />

            {/* Drawer */}
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 bottom-0 z-50 w-72 gradient-card border-l border-border/50 p-3 shadow-xl lg:hidden overflow-y-auto"
            >
              {/* Close Button */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {/* <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white shadow-glow"> */}
                    {/* <Video className="h-4 w-4 text-primary-foreground" /> */}
                    <img src="/logo-n.png" alt="NeuroClip" className="h-16 w-16 scale-150 object-contain" />
                  {/* </div> */}
                  {/* <span className="text-lg font-bold">NeuroClip</span> */}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setDrawerOpen(false)}
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              {/* Mobile Navigation */}
              <nav className="flex flex-col space-y-2">
                {navItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    onClick={() => setDrawerOpen(false)}
                    className={({ isActive }) =>
                      cn(
                        'flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-all hover:bg-muted',
                        isActive && 'bg-muted text-primary'
                      )
                    }
                  >
                    <item.icon className="h-5 w-5" />
                    {item.label}
                  </NavLink>
                ))}
              </nav>

              {/* User Section */}
              <div className="mt-auto pt-6 border-t border-border/50 space-y-4">
                <div className="px-4 py-2 rounded-lg bg-muted/50">
                  <p className="text-xs text-muted-foreground mb-1">Signed in as</p>
                  <p className="text-sm font-medium truncate">{user?.email}</p>
                </div>
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={handleSignOut}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign Out
                </Button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}
