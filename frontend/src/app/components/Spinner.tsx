
export function Spinner() {
    return (
      <div className="flex justify-center items-center h-screen flex-col">
        <div className="border-t-8 border-green-500 rounded-full animate-spin w-52 h-52"></div>
        <p className="mt-8 text-2xl font-mono">Analysis loading, this may take a few minutes...</p>
        <style jsx>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          .animate-spin {
            animation: spin 2s linear infinite;
          }
        `}</style>
      </div>
    );
  }
  
