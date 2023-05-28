import Image from 'next/image'

export default function Upload() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify- dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
          <a
            className="pointer-events-none flex place-items-center gap-3 p-8 lg:pointer-events-auto lg:p-0"
            href="/"
            target=""
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