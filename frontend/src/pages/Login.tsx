import { useState } from 'react';
import { useLocation } from 'wouter';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Eye, EyeOff, Loader2, ArrowLeft, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function Login() {
    const [, setLocation] = useLocation();
    const { login, loginWithGoogle, isLoading: authLoading } = useAuth();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [step, setStep] = useState<'email' | 'password'>('email');

    const handleEmailContinue = () => {
        if (!email) {
            setError('Please enter your email');
            return;
        }
        setError('');
        setStep('password');
    };

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!password) {
            setError('Please enter your password');
            return;
        }

        setIsLoading(true);
        setError('');

        try {
            await login(email, password);
            setLocation('/');
        } catch (err: any) {
            setError(err.message || 'Login failed');
        } finally {
            setIsLoading(false);
        }
    };

    if (authLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-background px-4 animate-in fade-in duration-300">
            <div className="w-full max-w-[400px] flex flex-col items-center">
                <button
                    onClick={() => setLocation('/')}
                    className="absolute top-6 left-6 flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors duration-200"
                >
                    <ArrowLeft className="h-5 w-5" />
                    <span className="text-sm font-medium">Back</span>
                </button>

                <div className="mb-8 animate-in zoom-in-50 duration-300">
                    <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center">
                        <Sparkles className="w-7 h-7 text-[#00FFFF]" />
                    </div>
                </div>

                <h1 className="text-2xl font-semibold text-foreground mb-8 animate-in slide-in-from-bottom-2 duration-300">Welcome back</h1>

                <form onSubmit={step === 'email' ? (e) => { e.preventDefault(); handleEmailContinue(); } : handleLogin} className="w-full space-y-4">
                    <div className={cn(
                        "transition-all duration-300 ease-out",
                        step === 'email' ? "opacity-100 translate-y-0" : "opacity-0 h-0 overflow-hidden -translate-y-2"
                    )}>
                        {step === 'email' && (
                            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                                <div className="space-y-2">
                                    <Label htmlFor="email" className="text-sm text-muted-foreground">Email address</Label>
                                    <Input
                                        id="email"
                                        type="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="you@example.com"
                                        className="h-12 bg-card border-border focus:border-primary transition-all duration-200 focus:ring-2 focus:ring-primary/20"
                                        autoFocus
                                    />
                                </div>

                                {error && <p className="text-sm text-destructive animate-in fade-in duration-200">{error}</p>}

                                <Button type="submit" className="w-full h-12 bg-primary hover:bg-primary/90 text-primary-foreground font-medium transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]">
                                    Continue
                                </Button>
                            </div>
                        )}
                    </div>

                    {step === 'password' && (
                        <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                            <div className="flex items-center justify-between px-3 py-3 bg-card border border-border rounded-lg transition-all duration-200 hover:border-primary/30">
                                <span className="text-foreground">{email}</span>
                                <button type="button" onClick={() => setStep('email')} className="text-primary hover:underline text-sm transition-colors duration-200">
                                    Edit
                                </button>
                            </div>

                            <div className="space-y-2 animate-in fade-in slide-in-from-bottom-2 duration-300 delay-100">
                                <Label htmlFor="password" className="text-sm text-muted-foreground">Password</Label>
                                <div className="relative">
                                    <Input
                                        id="password"
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="Password"
                                        className="h-12 bg-card border-border focus:border-primary pr-10 transition-all duration-200 focus:ring-2 focus:ring-primary/20"
                                        autoFocus
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors duration-200"
                                    >
                                        {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                    </button>
                                </div>
                            </div>

                            {error && <p className="text-sm text-destructive animate-in fade-in duration-200">{error}</p>}

                            <Button type="submit" disabled={isLoading} className="w-full h-12 bg-primary hover:bg-primary/90 text-primary-foreground font-medium transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]">
                                {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : 'Continue'}
                            </Button>
                        </div>
                    )}
                </form>

                <p className="mt-4 text-sm text-muted-foreground animate-in fade-in duration-300 delay-150">
                    Don't have an account?{' '}
                    <button onClick={() => setLocation('/signup')} className="text-primary hover:underline transition-colors duration-200">
                        Sign up
                    </button>
                </p>

                <div className="relative w-full my-6">
                    <div className="absolute inset-0 flex items-center">
                        <span className="w-full border-t border-border" />
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                        <span className="bg-background px-2 text-muted-foreground">OR</span>
                    </div>
                </div>

                <Button
                    variant="outline"
                    onClick={loginWithGoogle}
                    className="w-full h-12 border-border hover:bg-card transition-all duration-200 hover:scale-[1.01] active:scale-[0.99]"
                >
                    <svg className="w-5 h-5 mr-3" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                    </svg>
                    Continue with Google
                </Button>

                <p className="mt-8 text-xs text-muted-foreground text-center">
                    <a href="#" className="hover:underline transition-colors duration-200">Terms of use</a>
                    {' | '}
                    <a href="#" className="hover:underline transition-colors duration-200">Privacy policy</a>
                </p>
            </div>
        </div>
    );
}
