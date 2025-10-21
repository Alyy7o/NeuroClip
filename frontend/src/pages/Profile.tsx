import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { User, Mail, Save, Edit } from 'lucide-react';
import { DashboardLayout } from '@/components/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useAuth } from '@/contexts/AuthContext';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';
import { z } from 'zod';

const profileSchema = z.object({
  fullName: z.string()
    .trim()
    .min(1, { message: "Name is required" })
    .max(100, { message: "Name must be less than 100 characters" })
    .regex(/^[a-zA-Z\s'-]+$/, { message: "Name can only contain letters, spaces, hyphens and apostrophes" })
});

export default function Profile() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [profile, setProfile] = useState({
    full_name: '',
    email: user?.email || '',
  });

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user?.id)
        .single();

      if (error) throw error;
      
      if (data) {
        setProfile({
          full_name: data.full_name || '',
          email: data.email || user?.email || '',
        });
      }
    } catch (error) {
      console.error('Error fetching profile:', error);
    }
  };

  const handleSave = async () => {
    if (!user) return;
    
    setLoading(true);
    try {
      const result = profileSchema.safeParse({
        fullName: profile.full_name
      });

      if (!result.success) {
        toast({
          title: 'Validation Error',
          description: result.error.errors[0].message,
          variant: 'destructive'
        });
        setLoading(false);
        return;
      }

      const validated = result.data;

      const { error } = await supabase
        .from('profiles')
        .update({ 
          full_name: validated.fullName
        })
        .eq('id', user.id);

      if (error) throw error;

      toast({
        title: 'Success',
        description: 'Profile updated successfully'
      });
      setIsEditing(false);
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to update profile',
        variant: 'destructive'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-2xl mx-auto space-y-6"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg">
            <User className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold">Profile</h1>
            <p className="text-sm sm:text-base text-muted-foreground">
              Manage your account information
            </p>
          </div>
        </div>

        <Card className="gradient-card border-border/50">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Personal Information</CardTitle>
                <CardDescription>Update your profile details</CardDescription>
              </div>
              {!isEditing && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsEditing(true)}
                >
                  <Edit className="h-4 w-4 mr-2" />
                  Edit
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  value={profile.email}
                  disabled
                  className="pl-10"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Email cannot be changed
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="full_name">Full Name</Label>
              <Input
                id="full_name"
                type="text"
                value={profile.full_name}
                onChange={(e) =>
                  setProfile({ ...profile, full_name: e.target.value })
                }
                disabled={!isEditing}
                placeholder="Enter your full name"
              />
            </div>

            {isEditing && (
              <div className="flex gap-2 pt-4">
                <Button
                  onClick={handleSave}
                  disabled={loading}
                  className="flex-1 gradient-primary"
                >
                  <Save className="h-4 w-4 mr-2" />
                  {loading ? 'Saving...' : 'Save Changes'}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setIsEditing(false);
                    fetchProfile();
                  }}
                  disabled={loading}
                >
                  Cancel
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="gradient-card border-border/50">
          <CardHeader>
            <CardTitle>Account Statistics</CardTitle>
            <CardDescription>Your account activity</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground mb-1">Member Since</p>
                <p className="text-lg font-semibold">
                  {user?.created_at
                    ? new Date(user.created_at).toLocaleDateString('en-US', {
                        month: 'short',
                        year: 'numeric',
                      })
                    : 'N/A'}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground mb-1">Account ID</p>
                <p className="text-xs font-mono break-all">
                  {user?.id.substring(0, 8)}...
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </DashboardLayout>
  );
}
