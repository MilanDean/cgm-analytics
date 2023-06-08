
export default function TopNav() {
    
    return (
        <div className="mb-32 grid text-center lg:mb-0 lg:grid-cols-3 lg:text-left">
            <a
                href="/"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                target=""
                rel="noopener noreferrer"
            >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                Home{' '}
                <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                Navigate back to the Home page.
                </p>
            </a>

            <a
                href="/team"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                target=""
                rel="noopener noreferrer"
            >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                Team{' '}
                <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                Meet the the people behind the insights.
                </p>
            </a>

            <a
                href="/upload"
                className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800 hover:dark:bg-opacity-30"
                target=""
                rel="noopener noreferrer"
            >
                <h2 className={`mb-3 text-2xl font-semibold`}>
                Upload{' '}
                <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                    -&gt;
                </span>
                </h2>
                <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                Upload data to determine what food was consumed during CGM.
                </p>
            </a>
            
        </div>
    )
}