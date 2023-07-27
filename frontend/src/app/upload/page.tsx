'use client';

import { ChangeEvent, useState, useEffect } from 'react';
import { useRouter } from "next/navigation";
import axios from 'axios';
import TopNav from '../components/TopNav';
import UserAgreement from '../components/UserAgreement';

export default function Upload() {
  const [accepted, setAccepted] = useState(false);
  const [open, setOpen] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [age, setAge] = useState('');
  const [showButton, setShowButton] = useState(false);
  const router = useRouter();


  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file) {
      return;
    }

    if (!accepted) {
      setOpen(true);
      return;
    }

    const formData = new FormData();
    formData.append('file', file, file.name);

    localStorage.setItem('uploadedFilename', file.name);

    try {
      // Send the file to the FastAPI backend for processing
      const response = await axios.post('https://api.nutrinet-ai.com/api/analysis', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.status === 200 && accepted) {
        // Redirect to the analysis page to load the data we've just uploaded
        console.log('Successfully sent payload to FastAPI.');
        router.push('/analysis');
      }

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

  useEffect(() => {
    // Check if the user has already accepted the agreement
    if (accepted) {
      setShowButton(true);
    }
  }, [accepted]);

  return (
    <div className="flex flex-col h-screen">
      <div className="mt-4 mb-auto">
        <TopNav />
      </div>
      <div className="flex flex-col items-center justify-center mb-auto">
        <UserAgreement isOpen={open} setOpen={setOpen} setAccepted={setAccepted} />
        <form className="w-full max-w-3xl p-8 rounded-lg shadow-lg bg-transparent" onSubmit={handleSubmit}>
          <h1 className="text-3xl text-gray-600 font-bold text-center mb-8">CGM Data Upload</h1>
          <hr className="mb-8" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <input
              id="firstName"
              type="text"
              className="w-full p-2 rounded border border-gray-300"
              placeholder="First Name"
              value={firstName}
              onChange={e => setFirstName(e.target.value)}
            />
            <input
              id="lastName"
              type="text"
              className="w-full p-2 rounded border border-gray-300"
              placeholder="Last Name"
              value={lastName}
              onChange={e => setLastName(e.target.value)}
            />
          </div>
          <input
            id="age"
            type="number"
            className="w-full p-2 rounded border border-gray-300 mb-8"
            placeholder="Age"
            value={age}
            onChange={e => setAge(e.target.value)}
          />
          <div className="mb-4">
            <label htmlFor="fileInput" className="text-gray-600 font-bold block mb-2">
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
            {showButton && (
              <button
                type="submit"
                className="bg-green-600 text-white px-4 py-2 scale-125 rounded hover:bg-green-900"
                disabled={!accepted}
              >
                Upload
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  )};
