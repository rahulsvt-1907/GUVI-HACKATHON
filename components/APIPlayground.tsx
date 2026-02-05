
import React from 'react';

export const APIPlayground: React.FC = () => {
  const curlExample = `curl -X POST https://api.voxcheck.ai/v1/voice-detection \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: sk_live_1754586413785_019d3048-2158-7a8b-8497-d4488543595d" \\
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1..."
  }'`;

  const responseExample = `{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.94,
  "explanation": "High rhythmic consistency and synthetic frequency distribution noted."
}`;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h2 className="text-lg font-bold text-slate-900">Developer API</h2>
          <p className="text-sm text-slate-500 mt-1">Integrate voice authenticity detection into your enterprise workflow.</p>
        </div>

        <div className="p-8 space-y-8">
          <section className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Endpoint Specification</h3>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-green-600 text-white rounded font-black text-xs">POST</span>
              <code className="text-sm font-mono text-slate-700 bg-slate-100 px-3 py-1 rounded-lg flex-1">
                /v1/voice-detection
              </code>
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <section className="space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Request Body</h3>
              <div className="bg-slate-900 rounded-xl p-4 overflow-x-auto">
                <pre className="text-xs font-mono text-indigo-300">
                  {`{
  "language": "Tamil" | "English" | "Hindi" | "Malayalam" | "Telugu",
  "audioFormat": "mp3",
  "audioBase64": "string (Base64 MP3)"
}`}
                </pre>
              </div>
            </section>

            <section className="space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Authentication</h3>
              <div className="p-4 bg-amber-50 border border-amber-100 rounded-xl">
                <p className="text-xs font-bold text-amber-800 mb-2">Required Header:</p>
                <code className="text-xs font-mono bg-white px-2 py-1 border border-amber-200 rounded block">
                  x-api-key: YOUR_SECRET_KEY
                </code>
                <p className="text-[10px] text-amber-600 mt-2 italic">
                  * Note: For this demo, the frontend uses an injected API key via Google Generative AI.
                </p>
              </div>
            </section>
          </div>

          <section className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">cURL Integration</h3>
            <div className="relative">
              <div className="bg-slate-900 rounded-xl p-6 overflow-x-auto">
                <pre className="text-xs font-mono text-slate-300 leading-relaxed">
                  {curlExample}
                </pre>
              </div>
              <button
                onClick={() => navigator.clipboard.writeText(curlExample)}
                className="absolute top-4 right-4 text-slate-500 hover:text-white transition-colors"
                title="Copy to clipboard"
              >
                <i className="fas fa-copy"></i>
              </button>
            </div>
          </section>

          <section className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Success Response</h3>
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
              <pre className="text-xs font-mono text-slate-700">
                {responseExample}
              </pre>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
