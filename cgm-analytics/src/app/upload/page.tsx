import Image from 'next/image'
import TopNav from '../components/TopNav'

export default function Upload() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-12 m-2">

      <TopNav />
      <div className="flex flex-col items-center justify-center">
        <form className="w-full max-w-lg p-8 rounded-lg shadow-lg bg-transparent border-2 border-black">
          <h2 className="text-black text-2xl font-semibold mb-6">Upload File</h2>
          <div className="mb-4">
            <label htmlFor="fileInput" className="text-black font-bold">
              Select a file to upload:
            </label>
            <input
              type="file"
              id="fileInput"
              className="w-full border border-gray-300 p-2 rounded text-black bg-transparent"
            />
          </div>
          {/* Add additional form fields here if needed */}
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Upload
          </button>
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