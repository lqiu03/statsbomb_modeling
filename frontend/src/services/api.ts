const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatRequest {
  query: string;
  session_id?: string;
}

export interface ChatResponse {
  response: string;
  sources: Array<{
    title: string;
    url: string;
    content: string;
  }>;
  session_id: string;
}

export interface Competition {
  competition_id: number;
  competition_name: string;
  season_name: string;
  country_name: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getCompetitions(): Promise<Competition[]> {
    const response = await fetch(`${this.baseUrl}/competitions`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch competitions: ${response.statusText}`);
    }

    return response.json();
  }

  async search(query: string, filters?: Record<string, any>): Promise<any[]> {
    const params = new URLSearchParams({ query });
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value));
        }
      });
    }

    const response = await fetch(`${this.baseUrl}/search?${params}`);
    
    if (!response.ok) {
      throw new Error(`Search request failed: ${response.statusText}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiService = new ApiService();
