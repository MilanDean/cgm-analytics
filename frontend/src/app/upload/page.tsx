'use client';
import { ChangeEvent, useState } from 'react';
import axios from 'axios';
import Image from 'next/image';
import TopNav from '../components/TopNav';

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append('file', file, file.name);

    try {
      // Send the file to the FastAPI backend for processing
      const response = await axios.post('http://127.0.0.1:8000/api/analysis', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];

    // Validate file type
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
    } else {
      setFile(null);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-12 m-2">

      <TopNav />
      <div className="flex flex-col items-center justify-center">
        <form className="w-full max-w-lg p-8 rounded-lg shadow-lg bg-transparent border-2 border-black" onSubmit={handleSubmit}>

          <div className="p-6 mb-4">
            <label htmlFor="fileInput" className="text-black font-bold">
              Select a file to upload:
            </label>
            <input
              type="file"
              id="fileInput"
              accept=".csv"
              className="w-full border border-gray-300 p-2 rounded text-black bg-transparent"
              onChange={handleFileChange}
            />
          </div>

          <div className='flex items-center justify-center py-3'>
            <button
              type="submit"
              className="bg-black text-white px-4 py-2 scale-125 rounded hover:bg-blue-600"
            >
              Upload
            </button>
          </div>

        </form>
      </div>

      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify- dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
          <a
            className="pointer-events-none flex place-items-center gap-3 p-8 lg:pointer-events-auto lg:p-0"
            href="https://www.ischool.berkeley.edu/"
            target="/"
            rel="noopener noreferrer"
          >
            By:{' '}
            <Image
              src="/berkeleyischool-logo.svg"
              alt="UC Berkeley"
              className="dark:invert"
              width={150}
              height={18}
              priority
            />
          </a>
        </div>
      </div>

    </main>
  )
}