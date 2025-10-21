# NeuroClip - Professional Video Processing Platform

A beautiful, modern Progressive Web App for video processing built with React, TypeScript, and Lovable Cloud.

## Features

### 🎬 Three Core Modules
- **Video Summarization**: AI-powered extraction of key highlights based on custom queries
- **Video Blurring**: Intelligent blurring of faces, objects, or sensitive content
- **Video Compression**: Smart compression maintaining quality while reducing file size

### ✨ Key Highlights
- 🎨 Beautiful dual-theme design (light/dark mode)
- 📱 Fully responsive, mobile-first interface
- ⚡ Smooth animations with Framer Motion
- 🔐 Secure authentication via Lovable Cloud
- 📊 Complete processing history tracking
- 🎯 Modern, gradient-based UI with shadcn components

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite
- **UI**: Tailwind CSS, shadcn/ui, Framer Motion
- **Backend**: Lovable Cloud (Supabase)
- **Database**: PostgreSQL
- **Authentication**: Supabase Auth

## Getting Started

### Prerequisites
- Node.js 18+ and npm

### Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd NeuroClip
```

2. Install dependencies
```bash
npm install
```

3. Start development server
```bash
npm run dev
```

The app will be available at `http://localhost:8080`

## Project Structure

```
src/
├── components/       # Reusable UI components
├── contexts/        # React contexts (Auth, Theme)
├── pages/           # Page components
├── hooks/           # Custom React hooks
└── integrations/    # Supabase integration

supabase/
└── functions/       # Edge functions (if needed)
```

## Features in Detail

### Authentication
- Email/password signup and login
- Secure session management
- Protected routes for authenticated users

### Video Processing
All three modules follow a consistent workflow:
1. Upload video file or provide URL
2. Add processing instructions/query (for summarization & blurring)
3. Process with real-time progress tracking
4. View results and download processed video

### Processing History
- Track all processing jobs
- View detailed job information
- Filter by module type
- See compression statistics

## Design System

The app uses a comprehensive design system with:
- Semantic color tokens for consistent theming
- Gradient overlays and glow effects
- Smooth transitions and micro-interactions
- Custom button variants and card styles

## Deployment

Deploy to Lovable's hosting:
1. Click "Share" → "Publish" in the Lovable editor
2. Your app will be live with automatic HTTPS

## Important Notes

⚠️ **Video Processing**: This frontend demonstrates the complete UI/UX for video processing. For production use, you'll need to integrate actual video processing services:
- FFmpeg-based backend server
- Cloud video processing APIs (AWS MediaConvert, Cloudinary, etc.)
- Custom edge functions for processing logic

## Contributing

This project was built with Lovable. To contribute:
1. Make changes in the Lovable editor
2. Changes auto-commit to the repository
3. Or clone and push changes from your IDE

## License

MIT License - feel free to use for your projects!

## Support

For questions or issues:
- Check [Lovable Docs](https://docs.lovable.dev)
- Visit [Lovable Community](https://discord.gg/lovable)
