export const config = { runtime: 'edge' };

export default async function handler(req) {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 });
  }

  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  
  const response = await fetch(`${backendUrl}/predict`, {
    method: 'POST',
    headers: {
      ...Object.fromEntries(req.headers),
      'ngrok-skip-browser-warning': '1',
    },
    body: req.body,
    duplex: 'half',
  });

  const data = await response.arrayBuffer();
  
  return new Response(data, {
    status: response.status,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
  });
}
